"""
Synthetic training benchmark for ZipDepth (no dataset / I/O required).

Measures VRAM usage, step time, throughput (samples/sec), and estimated
Colab compute-unit / time cost across batch sizes on L4, A100, or any CUDA GPU.

Usage:
    # Benchmark Base model with standard batch size (96) at 384x384
    python scripts/benchmark_train.py --variant base --batch-sizes 32 48 64 96 --height 384 --width 384

    # Benchmark with HDN loss and 512x512 resolution
    python scripts/benchmark_train.py --variant base --loss-type hdn --batch-sizes 16 32 64 --height 512 --width 512
"""

import gc
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm.auto import tqdm

from zipdepth.model.architecture import create_model
from zipdepth.loss import create_loss


def parse_args():
    parser = argparse.ArgumentParser(description="ZipDepth synthetic training benchmark")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["small", "base", "large", "giant"],
                        help="Model variant (default: base)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                        help="Explicit batch sizes to test (if omitted, auto-finds saturation batch size)")
    parser.add_argument("--start-bs", type=int, default=8,
                        help="Starting batch size for auto-saturation search (default: 8)")
    parser.add_argument("--sat-threshold", type=float, default=0.05,
                        help="Stop doubling when throughput gain is below this fraction (default: 0.05 = 5%%)")
    parser.add_argument("--height", type=int, default=384, help="Input height (default: 384)")
    parser.add_argument("--width", type=int, default=384, help="Input width (default: 384)")
    parser.add_argument("--loss-type", type=str, default="original",
                        choices=["original", "hdn"],
                        help="Loss type: original (ZipDepthLoss) or hdn (HDNDistillationLoss)")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Precision dtype (default: bfloat16)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Micro-batches per optimizer update (batch-sizes are micro-batch sizes; "
                             "throughput counts all samples per update)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps (default: 5)")
    parser.add_argument("--steps", type=int, default=20, help="Measured steps (default: 20)")
    parser.add_argument("--dataset-size", type=int, default=1_000_000,
                        help="Reference dataset size for epoch time estimate (default: 1,000,000)")
    parser.add_argument("--cu-per-hour", type=float, default=None,
                        help="Colab Compute Units/hour override (auto-detected for L4/A100/T4 if omitted)")
    return parser.parse_args()


def detect_colab_cu_per_hour(gpu_name: str) -> float:
    name = gpu_name.upper()
    if "A100" in name:
        return 5.4
    if "L4" in name:
        return 1.9
    if "V100" in name:
        return 5.0
    if "T4" in name:
        return 1.2
    return 0.0


def benchmark_batch_size(args, batch_size: int, device: torch.device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    use_amp = args.amp_dtype != "float32"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "float16"))

    model = create_model(variant=args.variant).to(device)
    model.train()
    criterion = create_loss(loss_type=args.loss_type).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    images = torch.randn(batch_size, 3, args.height, args.width, device=device)
    target = torch.rand(batch_size, args.height, args.width, device=device) * 10.0 + 0.1
    mask = torch.ones(batch_size, args.height, args.width, dtype=torch.bool, device=device)

    accum = max(1, args.grad_accum_steps)

    def step():
        # One optimizer update = `accum` micro-batches (mirrors ZipDepthTrainer)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(images)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                if pred.ndim == 4 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                loss, _ = criterion(pred, target, mask)
            loss.item()  # trainer syncs on loss every micro-batch
            loss = loss / accum if accum > 1 else loss
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    try:
        for _ in tqdm(range(args.warmup), desc=f"bs={batch_size} warmup", leave=False):
            step()
        torch.cuda.synchronize(device)

        times = []
        for _ in tqdm(range(args.steps), desc=f"bs={batch_size} bench", leave=False):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        mean_step_ms = (sum(times) / len(times)) * 1000.0
        samples_per_sec = batch_size * accum / (sum(times) / len(times))
        return {
            "batch_size": batch_size,
            "oom": False,
            "peak_vram_gb": peak_vram_gb,
            "step_ms": mean_step_ms,
            "samples_per_sec": samples_per_sec,
        }
    except torch.cuda.OutOfMemoryError:
        return {
            "batch_size": batch_size,
            "oom": True,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "step_ms": 0.0,
            "samples_per_sec": 0.0,
        }
    finally:
        del model, criterion, optimizer, images, target, mask
        gc.collect()
        torch.cuda.empty_cache()


def find_saturation_batch_size(args, device: torch.device):
    """Double batch size until throughput saturates (< sat_threshold gain) or OOM,
    with a midpoint check if OOM occurs before saturation."""
    results = []
    bs = args.start_bs
    prev_res = None
    last_ok_bs = None
    oom_bs = None

    while True:
        res = benchmark_batch_size(args, bs, device)
        results.append(res)
        if res["oom"]:
            oom_bs = bs
            break
        last_ok_bs = bs
        if prev_res is not None:
            gain = (res["samples_per_sec"] - prev_res["samples_per_sec"]) / prev_res["samples_per_sec"]
            if gain < args.sat_threshold:
                break
        prev_res = res
        bs *= 2

    # If we hit OOM before saturating, check midpoint between last_ok_bs and oom_bs
    if oom_bs is not None and last_ok_bs is not None and (oom_bs - last_ok_bs) > 8:
        mid_bs = ((last_ok_bs + oom_bs) // 2 // 8) * 8
        if mid_bs > last_ok_bs:
            mid_res = benchmark_batch_size(args, mid_bs, device)
            results.append(mid_res)
            results.sort(key=lambda r: r["batch_size"])

    return results


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Run this on a GPU runtime (e.g., Colab L4 or A100).")
        sys.exit(1)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name
    total_vram_gb = props.total_memory / (1024 ** 3)
    cu_per_hr = args.cu_per_hour if args.cu_per_hour is not None else detect_colab_cu_per_hour(gpu_name)

    print("=" * 78)
    print(f"ZipDepth Synthetic Training Benchmark")
    print(f"GPU: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    print(f"Model: {args.variant} | Resolution: {args.height}x{args.width} | "
          f"Loss: {args.loss_type} | AMP: {args.amp_dtype} | "
          f"Accum: {args.grad_accum_steps} (step time = 1 optimizer update)")
    print(f"Reference dataset size: {args.dataset_size:,} samples"
          + (f" | Est. Colab rate: {cu_per_hr:.1f} CU/hr" if cu_per_hr > 0 else ""))
    print("=" * 78)

    if args.batch_sizes is not None:
        results = [benchmark_batch_size(args, bs, device) for bs in args.batch_sizes]
    else:
        results = find_saturation_batch_size(args, device)

    print()
    header = f"{'BS':>4} | {'Status':>7} | {'Peak VRAM':>10} | {'Step (ms)':>10} | {'Samp/sec':>9} | {'Hrs/Epoch':>9}"
    if cu_per_hr > 0:
        header += f" | {'CU/Epoch':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["oom"]:
            line = f"{r['batch_size']:>4} | {'OOM':>7} | {r['peak_vram_gb']:>8.2f}GB | {'-':>10} | {'-':>9} | {'-':>9}"
            if cu_per_hr > 0:
                line += f" | {'-':>8}"
        else:
            hrs = (args.dataset_size / r["samples_per_sec"]) / 3600.0
            line = (f"{r['batch_size']:>4} | {'OK':>7} | {r['peak_vram_gb']:>8.2f}GB | "
                    f"{r['step_ms']:>10.1f} | {r['samples_per_sec']:>9.1f} | {hrs:>9.2f}")
            if cu_per_hr > 0:
                line += f" | {hrs * cu_per_hr:>8.1f}"
        print(line)

    ok_results = [r for r in results if not r["oom"]]
    if ok_results:
        best = max(ok_results, key=lambda r: r["samples_per_sec"])
        hrs = (args.dataset_size / best["samples_per_sec"]) / 3600.0
        msg = (f"Recommended (max throughput): BS={best['batch_size']} | "
               f"{best['samples_per_sec']:.1f} samp/s | {best['peak_vram_gb']:.2f} GB VRAM | "
               f"{hrs:.2f} hrs/epoch")
        if cu_per_hr > 0:
            msg += f" | {hrs * cu_per_hr:.1f} CU/epoch"
        print(f"\n{msg}")
    print()


if __name__ == "__main__":
    main()
