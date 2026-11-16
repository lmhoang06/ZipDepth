"""
Loss function parity verification and speed/memory benchmark for ZipDepth.

Compares the optimized Triton/PyTorch losses (`ZipDepthLoss`, `HDNDistillationLoss`)
against the original PyTorch baseline implementations across:
  - Forward & backward numerical parity (dense, sparse, dp/dr, bf16/fp16/fp32, odd dims, CPU)
  - Colab-style AMP + GradScaler student smoke tests
  - Stage 1/2/3 latency (ms) and peak extra VRAM (MB)

Usage:
    python scripts/benchmark_losses.py
    python scripts/benchmark_losses.py --batch-size 384 --height 256 --width 256 --amp-dtype bfloat16
    python scripts/benchmark_losses.py --train-step --batch-size 128
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipdepth.loss import HDNDistillationLoss, ZipDepthLoss, create_loss
from zipdepth.model import create_model


# ---------------------------------------------------------------------------
# Baseline Reference Implementations (pre-optimization PyTorch behavior)
# ---------------------------------------------------------------------------

class _BaselineZipDepthLoss(nn.Module):
    def __init__(
        self,
        alpha_ssi: float = 1.0,
        alpha_grad: float = 2.0,
        grad_scales: int = 4,
    ) -> None:
        super().__init__()
        self.alpha: Dict[str, float] = {"ssi": alpha_ssi, "grad": alpha_grad}
        self.grad_scales = grad_scales

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.float32)

        p_hat, t_hat = self._normalize(pred, target, mask)
        ssi_loss = self._ssi_loss(p_hat, t_hat, mask)
        grad_loss = self._gradient_loss(p_hat, t_hat, mask)
        total = self.alpha["ssi"] * ssi_loss + self.alpha["grad"] * grad_loss
        return total, {"ssi": ssi_loss.detach().item(), "grad": grad_loss.detach().item()}

    def _normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target, mask = self._squeeze(pred, target, mask)
        B = pred.shape[0]
        pred_nan = pred.clone()
        tgt_nan = target.clone()
        pred_nan[mask < 0.5] = float("nan")
        tgt_nan[mask < 0.5] = float("nan")
        p_med = torch.nanmedian(pred_nan.view(B, -1), dim=1).values.view(B, 1, 1)
        t_med = torch.nanmedian(tgt_nan.view(B, -1), dim=1).values.view(B, 1, 1)
        M = mask.sum(dim=(1, 2), keepdim=True) + 1e-8
        p_mad = ((pred - p_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M
        t_mad = ((target - t_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M
        return (pred - p_med) / p_mad.clamp(min=1e-6), (target - t_med) / t_mad.clamp(min=1e-6)

    def _ssi_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred, target, mask = self._squeeze(pred, target, mask)
        return ((pred - target).abs() * mask).sum() / (mask.sum() + 1e-8)

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred, target, mask = self._squeeze(pred, target, mask)
        total: torch.Tensor = pred.new_zeros(())
        for scale in range(self.grad_scales):
            k = 2 ** scale
            if k > 1:
                p = F.avg_pool2d(pred.unsqueeze(1), k).squeeze(1)
                t = F.avg_pool2d(target.unsqueeze(1), k).squeeze(1)
                m = (F.avg_pool2d(mask.unsqueeze(1), k).squeeze(1) > 0.99).float()
            else:
                p, t, m = pred, target, mask
            p_dx, p_dy = p[:, :, 1:] - p[:, :, :-1], p[:, 1:, :] - p[:, :-1, :]
            t_dx, t_dy = t[:, :, 1:] - t[:, :, :-1], t[:, 1:, :] - t[:, :-1, :]
            m_x, m_y = m[:, :, 1:] * m[:, :, :-1], m[:, 1:, :] * m[:, :-1, :]
            denom = m_x.sum() + m_y.sum() + 1e-8
            total = total + (((p_dx - t_dx).abs() * m_x).sum() + ((p_dy - t_dy).abs() * m_y).sum()) / denom
        return total / self.grad_scales

    @staticmethod
    def _squeeze(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)


class _BaselineHDNDistillationLoss(nn.Module):
    def __init__(
        self,
        alpha_ssi: float = 1.0,
        alpha_hdn: float = 1.0,
        alpha_grad: float = 2.0,
        alpha_normal: float = 0.5,
        grad_scales: int = 4,
        trim_ratio: float = 0.10,
        hdn_segments: Sequence[int] = (2, 4),
        hdn_mode: str = "dp",
        normal_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha: Dict[str, float] = {
            "ssi": float(alpha_ssi),
            "hdn": float(alpha_hdn),
            "grad": float(alpha_grad),
            "normal": float(alpha_normal),
        }
        self.grad_scales = int(grad_scales)
        self.trim_ratio = float(trim_ratio)
        self.hdn_segments = tuple(int(s) for s in hdn_segments)
        self.hdn_mode = str(hdn_mode).lower()
        self.normal_beta = float(normal_beta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred, target = self._squeeze(pred, target)
        dense_mode = mask is None
        if dense_mode:
            mask = torch.ones_like(target, dtype=torch.float32)
        else:
            (mask,) = self._squeeze(mask)
            mask = mask.float()

        p_hat, t_hat = self._normalize_global(pred, target, mask, dense_mode=dense_mode)
        ssi_loss = self._trimmed_ssi_loss(p_hat, t_hat, mask, dense_mode=dense_mode)
        if self.alpha["hdn"] > 0 and len(self.hdn_segments) > 0:
            hdn_loss = self._hybrid_depth_norm_loss(pred, target, mask)
        else:
            hdn_loss = torch.zeros_like(ssi_loss)
        grad_loss, normal_loss = self._gradient_and_normal_loss(p_hat, t_hat, mask)

        total = (
            self.alpha["ssi"] * ssi_loss
            + self.alpha["hdn"] * hdn_loss
            + self.alpha["grad"] * grad_loss
            + self.alpha["normal"] * normal_loss
        )
        return total, {
            "ssi": ssi_loss.detach().item(),
            "hdn": hdn_loss.detach().item(),
            "grad": grad_loss.detach().item(),
            "normal": normal_loss.detach().item(),
        }

    def _normalize_global(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        dense_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = pred.shape[0]
        if dense_mode:
            p_med = torch.median(pred.view(B, -1), dim=1).values.view(B, 1, 1)
            t_med = torch.median(target.view(B, -1), dim=1).values.view(B, 1, 1)
            p_mad = (pred - p_med).abs().mean(dim=(1, 2), keepdim=True)
            t_mad = (target - t_med).abs().mean(dim=(1, 2), keepdim=True)
        else:
            pred_nan = pred.masked_fill(mask < 0.5, float("nan"))
            tgt_nan = target.masked_fill(mask < 0.5, float("nan"))
            p_med = torch.nanmedian(pred_nan.view(B, -1), dim=1).values.view(B, 1, 1)
            t_med = torch.nanmedian(tgt_nan.view(B, -1), dim=1).values.view(B, 1, 1)
            M = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            p_mad = ((pred - p_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M
            t_mad = ((target - t_med).abs() * mask).sum(dim=(1, 2), keepdim=True) / M
        return (pred - p_med) / p_mad.clamp(min=1e-6), (target - t_med) / t_mad.clamp(min=1e-6)

    def _trimmed_ssi_loss(
        self,
        p_hat: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
        dense_mode: bool = True,
    ) -> torch.Tensor:
        B = p_hat.shape[0]
        diff = (p_hat - t_hat).abs()
        if self.trim_ratio <= 0.0:
            return (diff * mask).sum() / (mask.sum() + 1e-8)
        if dense_mode:
            flat_diff = diff.view(B, -1)
            num_keep = int(flat_diff.shape[1] * (1.0 - self.trim_ratio))
            kept_diff, _ = torch.topk(flat_diff, k=num_keep, dim=1, largest=False, sorted=False)
            return kept_diff.mean()
        diff_masked = diff.masked_fill(mask < 0.5, float("inf")).view(B, -1)
        sorted_diff, _ = torch.sort(diff_masked, dim=1)
        valid_counts = (mask.view(B, -1) >= 0.5).sum(dim=1)
        keep_counts = (valid_counts.float() * (1.0 - self.trim_ratio)).long().clamp(min=1)
        idx = torch.arange(diff_masked.shape[1], device=p_hat.device).unsqueeze(0)
        keep_mask = (idx < keep_counts.unsqueeze(1)).float()
        safe_diff = torch.where(torch.isfinite(sorted_diff), sorted_diff, torch.zeros_like(sorted_diff))
        return (safe_diff * keep_mask).sum() / (keep_counts.sum().float() + 1e-8)

    def _hybrid_depth_norm_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W = target.shape
        t_flat = target.view(B, -1)
        seg_masks: List[torch.Tensor] = []
        if self.hdn_mode == "dp":
            for S in self.hdn_segments:
                q_probs = torch.linspace(0.0, 1.0, steps=S + 1, device=target.device, dtype=target.dtype)
                q_vals = torch.quantile(t_flat, q_probs, dim=1).view(S + 1, B, 1, 1)
                for s in range(S):
                    low, high = q_vals[s], q_vals[s + 1]
                    m_u = (target >= low) & (target <= high) if s == 0 else (target > low) & (target <= high)
                    seg_masks.append(m_u.float() * mask)
        else:
            t_min = t_flat.min(dim=1).values.view(B, 1, 1)
            t_max = t_flat.max(dim=1).values.view(B, 1, 1)
            t_span = (t_max - t_min).clamp(min=1e-6)
            for S in self.hdn_segments:
                step = t_span / S
                for s in range(S):
                    low, high = t_min + s * step, t_min + (s + 1) * step
                    m_u = (target >= low) & (target <= high + 1e-5) if s == S - 1 else (target >= low) & (target < high)
                    seg_masks.append(m_u.float() * mask)

        M_u = torch.stack(seg_masks, dim=1).view(B, -1, H * W)
        G = M_u.shape[1]
        p_exp = pred.view(B, 1, H * W).expand(B, G, H * W)
        t_exp = target.view(B, 1, H * W).expand(B, G, H * W)
        p_nan = p_exp.masked_fill(M_u < 0.5, float("nan"))
        t_nan = t_exp.masked_fill(M_u < 0.5, float("nan"))
        p_med = torch.nanmedian(p_nan, dim=2, keepdim=True).values.nan_to_num(0.0)
        t_med = torch.nanmedian(t_nan, dim=2, keepdim=True).values.nan_to_num(0.0)
        counts = M_u.sum(dim=2, keepdim=True)
        valid_group = (counts > 16.0).float()
        safe_counts = counts.clamp(min=1.0)
        p_mad = ((p_exp - p_med).abs() * M_u).sum(dim=2, keepdim=True) / safe_counts
        t_mad = ((t_exp - t_med).abs() * M_u).sum(dim=2, keepdim=True) / safe_counts
        non_flat = (t_mad > 1e-5).float() * valid_group
        p_norm = (p_exp - p_med) / p_mad.clamp(min=1e-6)
        t_norm = (t_exp - t_med) / t_mad.clamp(min=1e-6)
        diff = (p_norm - t_norm).abs() * M_u * non_flat
        return diff.sum() / ((M_u * non_flat).sum() + 1e-8)

    def _gradient_and_normal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_grad: torch.Tensor = pred.new_zeros(())
        total_normal: torch.Tensor = pred.new_zeros(())
        for scale in range(self.grad_scales):
            k = 2 ** scale
            if k > 1:
                p = F.avg_pool2d(pred.unsqueeze(1), k).squeeze(1)
                t = F.avg_pool2d(target.unsqueeze(1), k).squeeze(1)
                m = (F.avg_pool2d(mask.unsqueeze(1), k).squeeze(1) > 0.99).float()
            else:
                p, t, m = pred, target, mask
            p_dx, p_dy = p[:, :, 1:] - p[:, :, :-1], p[:, 1:, :] - p[:, :-1, :]
            t_dx, t_dy = t[:, :, 1:] - t[:, :, :-1], t[:, 1:, :] - t[:, :-1, :]
            m_x, m_y = m[:, :, 1:] * m[:, :, :-1], m[:, 1:, :] * m[:, :-1, :]
            denom = m_x.sum() + m_y.sum() + 1e-8
            total_grad = total_grad + (((p_dx - t_dx).abs() * m_x).sum() + ((p_dy - t_dy).abs() * m_y).sum()) / denom
            if self.alpha["normal"] > 0:
                px, py = self.normal_beta * p_dx[:, :-1, :], self.normal_beta * p_dy[:, :, :-1]
                tx, ty = self.normal_beta * t_dx[:, :-1, :], self.normal_beta * t_dy[:, :, :-1]
                m_xy = m_x[:, :-1, :] * m_y[:, :, :-1]
                ones = torch.ones_like(px)
                n_pred = F.normalize(torch.stack([-px, -py, ones], dim=1), dim=1, eps=1e-6)
                n_tgt = F.normalize(torch.stack([-tx, -ty, ones], dim=1), dim=1, eps=1e-6)
                total_normal = total_normal + ((1.0 - (n_pred * n_tgt).sum(dim=1)) * m_xy).sum() / (m_xy.sum() + 1e-8)
        return total_grad / self.grad_scales, total_normal / self.grad_scales

    @staticmethod
    def _squeeze(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)


# ---------------------------------------------------------------------------
# Verification & Benchmark Utilities
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int,
    height: int,
    width: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = torch.randn(batch_size, 1, height, width, device=device, dtype=dtype, requires_grad=True)
    target = torch.rand(batch_size, 1, height, width, device=device, dtype=torch.float32) * 10.0
    if batch_size >= 2:
        target[0] = 0.0
        target[1:, :, : height // 3, :] = 0.0
    return pred, target


def _check_parity(
    name: str,
    l_ref: torch.Tensor,
    d_ref: Dict[str, float],
    g_ref: torch.Tensor,
    l_opt: torch.Tensor,
    d_opt: Dict[str, float],
    g_opt: torch.Tensor,
    tol_l: float = 1e-5,
    tol_g: float = 5e-5,
    tol_gm: float = 1e-6,
) -> bool:
    l_diff = (l_ref.float() - l_opt.float()).abs().item()
    max_k = max(abs(d_ref[k] - d_opt[k]) for k in d_ref)
    g_diff = (g_ref.float() - g_opt.float()).abs()
    g_max = g_diff.max().item()
    g_mean = g_diff.mean().item()
    ok = (
        torch.isfinite(l_opt).all().item()
        and torch.isfinite(g_opt).all().item()
        and l_diff <= tol_l
        and max_k <= tol_l
        and g_max <= tol_g
        and g_mean <= tol_gm
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name:42s} | loss_diff={l_diff:.2e} | dict_max={max_k:.2e} "
        f"| bwd_max={g_max:.2e} | bwd_mean={g_mean:.2e}"
    )
    return ok


def verify_losses(hdn_mode: str = "dp") -> None:
    print("\n=== Loss Numerical & Gradient Parity Suite ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    all_ok = True

    if device == "cuda":
        cases = [
            ("Zip dense fp32 128x128", _BaselineZipDepthLoss, "original", {}, 8, 128, 128, False, torch.float32, None),
            ("Zip dense odd 127x213", _BaselineZipDepthLoss, "original", {"grad_scales": 5}, 6, 127, 213, False, torch.float32, None),
            ("Zip sparse mask 128x128", _BaselineZipDepthLoss, "original", {}, 6, 128, 128, True, torch.float32, None),
            ("Zip fp16 pred + GradScaler-safe", _BaselineZipDepthLoss, "original", {}, 6, 128, 128, False, torch.float16, None),
            (f"HDN {hdn_mode} dense 128x128", _BaselineHDNDistillationLoss, "hdn", {"hdn_mode": hdn_mode}, 8, 128, 128, False, torch.float32, None),
            (f"HDN {hdn_mode} odd 127x213", _BaselineHDNDistillationLoss, "hdn", {"hdn_mode": hdn_mode, "hdn_segments": (2, 3, 4)}, 6, 127, 213, False, torch.float32, None),
            (f"HDN {hdn_mode} sparse mask 128x128", _BaselineHDNDistillationLoss, "hdn", {"hdn_mode": hdn_mode}, 6, 128, 128, True, torch.float32, None),
            (f"HDN {hdn_mode} bf16 autocast", _BaselineHDNDistillationLoss, "hdn", {"hdn_mode": hdn_mode}, 6, 128, 128, False, torch.float32, torch.bfloat16),
        ]
        for label, RefCls, loss_name, kw, B, H, W, sparse, dtype, amp_dt in cases:
            ref = RefCls(**kw).to(device)
            opt = create_loss(loss_name, **kw).to(device)
            p0, t = _make_batch(B, H, W, device=device, dtype=dtype)
            p1 = p0.detach().clone().requires_grad_(True)
            m = (torch.rand(B, 1, H, W, device=device) > 0.3).float() if sparse else None

            if amp_dt is not None:
                with torch.amp.autocast("cuda", dtype=amp_dt):
                    l0, d0 = ref(p0, t, m)
                    l1, d1 = opt(p1, t, m)
            else:
                l0, d0 = ref(p0.float() if dtype != torch.float32 else p0, t, m)
                l1, d1 = opt(p1, t, m)
            l0.backward()
            l1.backward()

            tol_l = 5e-3 if dtype != torch.float32 else 1e-5
            tol_g = 2e-2 if dtype != torch.float32 else 5e-5
            tol_gm = 5e-4 if dtype != torch.float32 else 1e-6
            all_ok &= _check_parity(label, l0, d0, p0.grad, l1, d1, p1.grad, tol_l, tol_g, tol_gm)

        # Edge cases: empty mask + alpha_normal=0 + T4/fp16 GradScaler simulation
        p_em = torch.ones(3, 1, 64, 64, device=device, requires_grad=True)
        t_em = torch.rand(3, 1, 64, 64, device=device)
        m_em = torch.ones(3, 1, 64, 64, device=device)
        m_em[0] = 0.0
        for label, crit in [
            ("Zip all-masked+const", ZipDepthLoss().to(device)),
            ("HDN all-masked+const+normal0", HDNDistillationLoss(alpha_normal=0.0).to(device)),
        ]:
            if p_em.grad is not None:
                p_em.grad = None
            l_em, d_em = crit(p_em, t_em, m_em)
            l_em.backward()
            ok = torch.isfinite(l_em).item() and torch.isfinite(p_em.grad).all().item() and all(math.isfinite(v) for v in d_em.values())
            print(f"[{'PASS' if ok else 'FAIL'}] {label:42s} | finite={ok}")
            all_ok &= ok

        # Colab T4 fp16 + GradScaler smoke test (loss grad finite + scaler step converges cleanly)
        model = create_model("small", pretrained=False).to(device)
        opt_adam = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(4, 3, 256, 256, device=device)
        t_s = torch.rand(4, 1, 256, 256, device=device) * 10.0
        scaler = torch.amp.GradScaler("cuda")
        opt_hdn = HDNDistillationLoss(hdn_mode=hdn_mode).to(device)
        gs_ok = True
        for _ in range(3):
            opt_adam.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                y = model(x)
                y.retain_grad()
                l_s, _ = opt_hdn(y, t_s)
            scaler.scale(l_s).backward()
            gs_ok &= torch.isfinite(l_s).item() and y.grad is not None and torch.isfinite(y.grad).all().item()
            scaler.step(opt_adam)
            scaler.update()
        gs_ok &= all(torch.isfinite(p).all().item() for p in model.parameters()) and gs_ok
        print(f"[{'PASS' if gs_ok else 'FAIL'}] {'Colab T4 fp16 + GradScaler smoke':42s} | finite={gs_ok}")
        all_ok &= gs_ok

    # CPU fallback check
    for label, RefCls, OptCls, kw in [
        ("CPU fallback Zip", _BaselineZipDepthLoss, ZipDepthLoss, {}),
        (f"CPU fallback HDN ({hdn_mode})", _BaselineHDNDistillationLoss, HDNDistillationLoss, {"hdn_mode": hdn_mode}),
    ]:
        p0, t = _make_batch(2, 48, 48, device="cpu")
        p1 = p0.detach().clone().requires_grad_(True)
        l0, d0 = RefCls(**kw)(p0, t)
        l0.backward()
        l1, d1 = OptCls(**kw)(p1, t)
        l1.backward()
        all_ok &= _check_parity(label, l0, d0, p0.grad, l1, d1, p1.grad)

    if not all_ok:
        raise AssertionError("Verification failed on one or more parity tests.")


def _measure_step(
    criterion: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    amp_dtype: Optional[torch.dtype],
    warmup: int,
    iters: int,
) -> Tuple[Optional[float], Optional[float]]:
    use_amp = amp_dtype is not None

    def step() -> None:
        if pred.grad is not None:
            pred.grad = None
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss, _ = criterion(pred, target)
        loss.backward()

    try:
        torch.cuda.synchronize()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000.0
        peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024.0 ** 2)
        return ms, peak_mb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def benchmark_losses(
    shapes: Sequence[Tuple[str, int, int, int]],
    amp_dtype: Optional[torch.dtype],
    hdn_mode: str,
    warmup: int,
    iters: int,
) -> None:
    print(f"\n=== Loss Forward+Backward Benchmark (GPU: {torch.cuda.get_device_name(0)}, amp={amp_dtype}) ===")
    for label, B, H, W in shapes:
        print(f"\n--- {label}: B={B}, H={H}, W={W} ---")
        torch.manual_seed(0)
        p = torch.randn(B, 1, H, W, device="cuda", dtype=torch.float32, requires_grad=True)
        t = torch.rand(B, 1, H, W, device="cuda", dtype=torch.float32) * 10.0
        t[:, :, : H // 3, :] = 0.0

        pairs: Sequence[Tuple[str, Type[nn.Module], Type[nn.Module], Dict[str, str]]] = [
            ("ZipDepthLoss", _BaselineZipDepthLoss, ZipDepthLoss, {}),
            ("HDNDistillationLoss", _BaselineHDNDistillationLoss, HDNDistillationLoss, {"hdn_mode": hdn_mode}),
        ]
        for name, RefCls, OptCls, kw in pairs:
            ms_ref, mem_ref = _measure_step(RefCls(**kw).cuda(), p, t, amp_dtype, warmup=min(2, warmup), iters=min(5, iters))
            torch.cuda.empty_cache()
            ms_opt, mem_opt = _measure_step(OptCls(**kw).cuda(), p, t, amp_dtype, warmup=warmup, iters=iters)
            torch.cuda.empty_cache()
            if ms_opt is None:
                print(f"  {name:22s}: OOM at B={B}, {H}x{W}")
            elif ms_ref is None:
                print(f"  {name:22s}: Orig OOM -> Opt {ms_opt:6.2f} ms ({mem_opt:7.1f} MB extra)")
            else:
                print(
                    f"  {name:22s}: Orig {ms_ref:7.2f} ms ({mem_ref:7.1f} MB) -> "
                    f"Opt {ms_opt:6.2f} ms ({mem_opt:7.1f} MB) | {ms_ref / ms_opt:5.2f}x faster, {mem_ref / max(mem_opt, 1e-3):4.1f}x less extra VRAM"
                )


def benchmark_train_step(
    batch_size: int,
    height: int,
    width: int,
    amp_dtype: Optional[torch.dtype],
    hdn_mode: str,
    iters: int,
) -> None:
    print(f"\n=== End-to-End ZipDepth-Small Train Step (B={batch_size}, {height}x{width}, amp={amp_dtype}) ===")
    device = "cuda"
    images = torch.rand(batch_size, 3, height, width, device=device)
    target = torch.rand(batch_size, 1, height, width, device=device) * 10.0

    for label, crit in [
        ("Orig Baseline Zip", _BaselineZipDepthLoss().to(device)),
        ("Opt ZipDepthLoss", ZipDepthLoss().to(device)),
        ("Orig Baseline HDN", _BaselineHDNDistillationLoss(hdn_mode=hdn_mode).to(device)),
        ("Opt HDNLoss", HDNDistillationLoss(hdn_mode=hdn_mode).to(device)),
    ]:
        model = create_model("small", pretrained=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(images)
                loss, _ = crit(pred, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        try:
            torch.cuda.synchronize()
            for _ in range(2):
                step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                step()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / iters
            print(f"  {label:20s}: {elapsed * 1000.0:7.2f} ms/step ({1.0 / elapsed:5.2f} it/s)")
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:20s}: OOM on this GPU at B={batch_size}")
        finally:
            del model, optimizer
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark & verify ZipDepth losses")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--hdn-mode", type=str, default="dp", choices=["dp", "dr"])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--train-step", action="store_true", help="Also benchmark ZipDepth-Small full fwd+bwd+opt step")
    args = parser.parse_args()

    if not args.no_verify:
        verify_losses(hdn_mode=args.hdn_mode)

    if not torch.cuda.is_available():
        print("\nCUDA unavailable; skipping GPU timing benchmarks.")
        return

    if args.amp_dtype == "auto":
        major, _ = torch.cuda.get_device_capability(0)
        amp_dtype: Optional[torch.dtype] = torch.bfloat16 if major >= 8 else torch.float16
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    if args.batch_size is not None or args.height is not None or args.width is not None:
        shapes = [(
            "Custom",
            args.batch_size or 128,
            args.height or 256,
            args.width or 256,
        )]
    else:
        shapes = [
            ("Stage 1", 384, 256, 256),
            ("Stage 2", 128, 384, 384),
            ("Stage 3", 64, 512, 512),
        ]

    benchmark_losses(shapes, amp_dtype=amp_dtype, hdn_mode=args.hdn_mode, warmup=args.warmup, iters=args.iters)

    if args.train_step:
        bs = args.batch_size or 64
        h = args.height or 256
        w = args.width or 256
        benchmark_train_step(bs, h, w, amp_dtype=amp_dtype, hdn_mode=args.hdn_mode, iters=max(5, args.iters // 2))


if __name__ == "__main__":
    main()
