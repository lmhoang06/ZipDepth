"""Fast Triton kernels and PyTorch fallbacks for ZipDepth and HDN loss functions."""

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = hasattr(tl, "histogram")
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def can_use_triton(tensor: torch.Tensor) -> bool:
    """Return True when Triton CUDA kernels can execute on the given device."""
    if not (HAS_TRITON and tensor.is_cuda):
        return False
    major, _ = torch.cuda.get_device_capability(tensor.device)
    return major >= 7


if HAS_TRITON:

    @triton.jit
    def _dense_median_kernel(
        x_ptr,
        out_val_ptr,
        out_idx_ptr,
        B,
        N,
        FIND_IDX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """4-pass radix-select lower median along dim=1 for dense [B, N] float tensors."""
        b = tl.program_id(0)
        base = b * N
        k_rem = (N - 1) // 2
        prefix = tl.zeros((), dtype=tl.uint32)
        mask_prefix = tl.zeros((), dtype=tl.uint32)
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        for pass_i in tl.static_range(4):
            shift = 24 - 8 * pass_i
            hist = tl.zeros([256], dtype=tl.int32)
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                val = tl.load(x_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                # Canonicalize -0.0 to +0.0 before bitcast ordering
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                match = in_b if pass_i == 0 else (in_b & ((key & mask_prefix) == prefix))
                byte_val = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
                h512 = tl.histogram(byte_val, 512)
                hist += tl.sum(tl.reshape(h512, [2, 256]) * keep_lo, axis=0)

            csum = tl.cumsum(hist, axis=0)
            bucket = tl.argmax((csum > k_rem).to(tl.int32), axis=0)
            k_rem -= tl.sum(tl.where(idx_256 == (bucket - 1), csum, 0), axis=0)
            prefix = prefix | (bucket.to(tl.uint32) << shift)
            mask_prefix = mask_prefix | (tl.full((), 0xFF, tl.uint32) << shift)

        inv_u = prefix ^ tl.where(
            (prefix >> 31) == 0,
            tl.full((), 0xFFFFFFFF, tl.uint32),
            tl.full((), 0x80000000, tl.uint32),
        )
        tl.store(out_val_ptr + b, inv_u.to(tl.float32, bitcast=True))

        if FIND_IDX:
            min_idx = N
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                val = tl.load(x_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                hit = in_b & (key == prefix)
                min_idx = tl.minimum(min_idx, tl.min(tl.where(hit, offs, N), axis=0))
            tl.store(out_idx_ptr + b, min_idx)

    @triton.jit
    def _masked_median_kernel(
        x_ptr,
        m_ptr,
        out_val_ptr,
        out_idx_ptr,
        B,
        N,
        FIND_IDX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """4-pass radix-select lower median over valid elements (mask > 0.5)."""
        b = tl.program_id(0)
        base = b * N

        valid_cnt = tl.zeros((), dtype=tl.int32)
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_b = offs < N
            mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            valid_cnt += tl.sum((in_b & (mv > 0.5)).to(tl.int32), axis=0)

        if valid_cnt <= 0:
            tl.store(out_val_ptr + b, 0.0)
            if FIND_IDX:
                tl.store(out_idx_ptr + b, N)
            return

        k_rem = (valid_cnt - 1) // 2
        prefix = tl.zeros((), dtype=tl.uint32)
        mask_prefix = tl.zeros((), dtype=tl.uint32)
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        for pass_i in tl.static_range(4):
            shift = 24 - 8 * pass_i
            hist = tl.zeros([256], dtype=tl.int32)
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                ok = in_b & (mv > 0.5)
                val = tl.load(x_ptr + base + offs, mask=ok, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                match = ok if pass_i == 0 else (ok & ((key & mask_prefix) == prefix))
                byte_val = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
                h512 = tl.histogram(byte_val, 512)
                hist += tl.sum(tl.reshape(h512, [2, 256]) * keep_lo, axis=0)

            csum = tl.cumsum(hist, axis=0)
            bucket = tl.argmax((csum > k_rem).to(tl.int32), axis=0)
            k_rem -= tl.sum(tl.where(idx_256 == (bucket - 1), csum, 0), axis=0)
            prefix = prefix | (bucket.to(tl.uint32) << shift)
            mask_prefix = mask_prefix | (tl.full((), 0xFF, tl.uint32) << shift)

        inv_u = prefix ^ tl.where(
            (prefix >> 31) == 0,
            tl.full((), 0xFFFFFFFF, tl.uint32),
            tl.full((), 0x80000000, tl.uint32),
        )
        tl.store(out_val_ptr + b, inv_u.to(tl.float32, bitcast=True))

        if FIND_IDX:
            min_idx = N
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                ok = in_b & (mv > 0.5)
                val = tl.load(x_ptr + base + offs, mask=ok, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                hit = ok & (key == prefix)
                min_idx = tl.minimum(min_idx, tl.min(tl.where(hit, offs, N), axis=0))
            tl.store(out_idx_ptr + b, min_idx)

    @triton.jit
    def _norm_mad_tile_kernel(
        p_ptr,
        t_ptr,
        m_ptr,
        p_med_ptr,
        t_med_ptr,
        out_p_part_ptr,
        out_t_part_ptr,
        out_m_part_ptr,
        B,
        N,
        num_tiles,
        HAS_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N
        base = b * N
        pm = tl.load(p_med_ptr + b).to(tl.float32)
        tm = tl.load(t_med_ptr + b).to(tl.float32)
        pv = tl.load(p_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        tv = tl.load(t_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        idx = b * num_tiles + tile
        if HAS_MASK:
            mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            w = tl.where(in_b, mv, 0.0)
            tl.store(out_p_part_ptr + idx, tl.sum(tl.abs((pv - pm) * w), axis=0))
            tl.store(out_t_part_ptr + idx, tl.sum(tl.abs((tv - tm) * w), axis=0))
            tl.store(out_m_part_ptr + idx, tl.sum(w, axis=0))
        else:
            tl.store(out_p_part_ptr + idx, tl.sum(tl.where(in_b, tl.abs(pv - pm), 0.0), axis=0))
            tl.store(out_t_part_ptr + idx, tl.sum(tl.where(in_b, tl.abs(tv - tm), 0.0), axis=0))

    @triton.jit
    def _norm_apply_fwd_kernel(
        p_ptr,
        t_ptr,
        p_med_ptr,
        t_med_ptr,
        p_mad_ptr,
        t_mad_ptr,
        out_phat_ptr,
        out_that_ptr,
        B,
        N,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N
        base = b * N

        pm = tl.load(p_med_ptr + b).to(tl.float32)
        tm = tl.load(t_med_ptr + b).to(tl.float32)
        ps = tl.maximum(tl.load(p_mad_ptr + b).to(tl.float32), 1e-6)
        ts = tl.maximum(tl.load(t_mad_ptr + b).to(tl.float32), 1e-6)

        pv = tl.load(p_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        tv = tl.load(t_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        tl.store(out_phat_ptr + base + offs, (pv - pm) / ps, mask=in_b)
        tl.store(out_that_ptr + base + offs, (tv - tm) / ts, mask=in_b)

    @triton.jit
    def _norm_bwd_tile_kernel(
        p_ptr,
        m_ptr,
        dphat_ptr,
        p_med_ptr,
        out_g1_ptr,
        out_g2_ptr,
        out_gs_ptr,
        B,
        N,
        num_tiles,
        HAS_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Reduce per-tile gradient stats without re-reading p_hat from HBM."""
        b = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N
        base = b * N

        pm = tl.load(p_med_ptr + b).to(tl.float32)
        pv = tl.load(p_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        dp = tl.load(dphat_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        diff = pv - pm
        sgn = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        if HAS_MASK:
            mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            sgn = sgn * tl.abs(mv)

        idx = b * num_tiles + tile
        tl.store(out_g1_ptr + idx, tl.sum(tl.where(in_b, dp, 0.0), axis=0))
        tl.store(out_g2_ptr + idx, tl.sum(tl.where(in_b, dp * diff, 0.0), axis=0))
        tl.store(out_gs_ptr + idx, tl.sum(tl.where(in_b, sgn, 0.0), axis=0))

    @triton.jit
    def _norm_bwd_reduce_kernel(
        g1_part_ptr,
        g2_part_ptr,
        gs_part_ptr,
        p_mad_ptr,
        denom_ptr,
        out_coeff_ptr,
        out_gmed_ptr,
        out_ps_ptr,
        B,
        num_tiles,
    ):
        b = tl.program_id(0)
        base = b * num_tiles
        g1 = tl.zeros((), dtype=tl.float32)
        g2 = tl.zeros((), dtype=tl.float32)
        gs = tl.zeros((), dtype=tl.float32)
        for t_i in range(num_tiles):
            g1 += tl.load(g1_part_ptr + base + t_i).to(tl.float32)
            g2 += tl.load(g2_part_ptr + base + t_i).to(tl.float32)
            gs += tl.load(gs_part_ptr + base + t_i).to(tl.float32)
        pmad = tl.load(p_mad_ptr + b).to(tl.float32)
        ps = tl.maximum(pmad, 1e-6)
        M = tl.load(denom_ptr + b).to(tl.float32)
        not_clamped = pmad >= 1e-6
        coeff = (g2 / (ps * M)) if not_clamped else 0.0
        gmed = (-g1 + (gs * coeff if not_clamped else 0.0)) / ps
        tl.store(out_coeff_ptr + b, coeff)
        tl.store(out_gmed_ptr + b, gmed)
        tl.store(out_ps_ptr + b, ps)

    @triton.jit
    def _norm_bwd_write_kernel(
        p_ptr,
        m_ptr,
        dphat_ptr,
        grad_p_ptr,
        p_med_ptr,
        p_idx_ptr,
        coeff_ptr,
        gmed_ptr,
        ps_ptr,
        B,
        N,
        HAS_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N
        base = b * N

        pm = tl.load(p_med_ptr + b).to(tl.float32)
        p_idx = tl.load(p_idx_ptr + b)
        coeff = tl.load(coeff_ptr + b).to(tl.float32)
        gmed = tl.load(gmed_ptr + b).to(tl.float32)
        ps = tl.load(ps_ptr + b).to(tl.float32)

        pv = tl.load(p_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        dp = tl.load(dphat_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        diff = pv - pm
        sgn = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        if HAS_MASK:
            mv = tl.load(m_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            sgn = sgn * tl.abs(mv)
        gp = (dp - coeff * sgn) / ps
        gp += tl.where(offs == p_idx, gmed, 0.0)
        tl.store(grad_p_ptr + base + offs, gp.to(grad_p_ptr.dtype.element_ty), mask=in_b)

    @triton.jit
    def _ssi_fwd_tile_kernel(
        ph_ptr,
        th_ptr,
        m_ptr,
        out_sum_ptr,
        out_m_ptr,
        N_total,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        in_b = offs < N_total
        pv = tl.load(ph_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
        tv = tl.load(th_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
        err = tl.abs(pv - tv)
        if HAS_MASK:
            mv = tl.load(m_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
            w = tl.where(in_b, mv, 0.0)
            tl.store(out_sum_ptr + pid, tl.sum(err * w, axis=0))
            tl.store(out_m_ptr + pid, tl.sum(w, axis=0))
        else:
            tl.store(out_sum_ptr + pid, tl.sum(tl.where(in_b, err, 0.0), axis=0))

    @triton.jit
    def _ssi_bwd_kernel(
        ph_ptr,
        th_ptr,
        m_ptr,
        grad_out_ptr,
        denom_ptr,
        grad_ph_ptr,
        N_total,
        N_total_f,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        in_b = offs < N_total
        denom = tl.load(denom_ptr).to(tl.float32) if HAS_MASK else N_total_f
        scale = tl.load(grad_out_ptr).to(tl.float32) / denom
        pv = tl.load(ph_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
        tv = tl.load(th_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
        d = pv - tv
        sgn = tl.where(d > 0.0, 1.0, tl.where(d < 0.0, -1.0, 0.0))
        if HAS_MASK:
            mv = tl.load(m_ptr + offs, mask=in_b, other=0.0).to(tl.float32)
            sgn = sgn * mv
        tl.store(grad_ph_ptr + offs, scale * sgn, mask=in_b)

    @triton.jit
    def _trimmed_ssi_stats_kernel(
        ph_ptr,
        th_ptr,
        out_sum_ptr,
        out_tau_ptr,
        out_eq_weight_ptr,
        B,
        N,
        num_keep,
        BLOCK_N: tl.constexpr,
    ):
        """Find num_keep-th smallest |ph - th| per sample and compute exact trimmed sum."""
        b = tl.program_id(0)
        base = b * N
        k_rem = num_keep - 1
        prefix = tl.zeros((), dtype=tl.uint32)
        mask_prefix = tl.zeros((), dtype=tl.uint32)
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        # Non-negative float32 bit patterns are monotonic after canonicalizing -0.0
        for pass_i in tl.static_range(4):
            shift = 24 - 8 * pass_i
            hist = tl.zeros([256], dtype=tl.int32)
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                pv = tl.load(ph_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                tv = tl.load(th_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                d = tl.abs(pv - tv)
                d = tl.where(d == 0.0, 0.0, d)
                key = d.to(tl.uint32, bitcast=True)
                match = in_b if pass_i == 0 else (in_b & ((key & mask_prefix) == prefix))
                byte_val = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
                h512 = tl.histogram(byte_val, 512)
                hist += tl.sum(tl.reshape(h512, [2, 256]) * keep_lo, axis=0)

            csum = tl.cumsum(hist, axis=0)
            bucket = tl.argmax((csum > k_rem).to(tl.int32), axis=0)
            k_rem -= tl.sum(tl.where(idx_256 == (bucket - 1), csum, 0), axis=0)
            prefix = prefix | (bucket.to(tl.uint32) << shift)
            mask_prefix = mask_prefix | (tl.full((), 0xFF, tl.uint32) << shift)

        tau = prefix.to(tl.float32, bitcast=True)

        sum_lt = tl.zeros((), dtype=tl.float32)
        cnt_lt = tl.zeros((), dtype=tl.int32)
        cnt_eq = tl.zeros((), dtype=tl.int32)
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_b = offs < N
            pv = tl.load(ph_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            tv = tl.load(th_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            d = tl.abs(pv - tv)
            lt = in_b & (d < tau)
            eq = in_b & (d == tau)
            sum_lt += tl.sum(tl.where(lt, d, 0.0), axis=0)
            cnt_lt += tl.sum(lt.to(tl.int32), axis=0)
            cnt_eq += tl.sum(eq.to(tl.int32), axis=0)

        rem = num_keep - cnt_lt
        row_sum = sum_lt + rem.to(tl.float32) * tau
        eq_w = tl.where(cnt_eq > 0, rem.to(tl.float32) / cnt_eq.to(tl.float32), 0.0)
        tl.store(out_sum_ptr + b, row_sum)
        tl.store(out_tau_ptr + b, tau)
        tl.store(out_eq_weight_ptr + b, eq_w)

    @triton.jit
    def _trimmed_ssi_bwd_kernel(
        ph_ptr,
        th_ptr,
        tau_ptr,
        eq_weight_ptr,
        grad_out_ptr,
        grad_ph_ptr,
        B,
        N,
        inv_denom,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N
        base = b * N

        scale = tl.load(grad_out_ptr).to(tl.float32) * inv_denom
        tau = tl.load(tau_ptr + b).to(tl.float32)
        eq_w = tl.load(eq_weight_ptr + b).to(tl.float32)
        pv = tl.load(ph_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        tv = tl.load(th_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        diff = pv - tv
        ad = tl.abs(diff)
        w = tl.where(ad < tau, 1.0, tl.where(ad == tau, eq_w, 0.0))
        sgn = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        tl.store(grad_ph_ptr + base + offs, scale * w * sgn, mask=in_b)

    @triton.jit
    def _sorted_seg_stats_kernel(
        p_sorted_ptr,
        t_sorted_ptr,
        m_sorted_ptr,
        starts_ptr,
        ends_ptr,
        out_p_med_ptr,
        out_t_med_ptr,
        out_p_mad_ptr,
        out_t_mad_ptr,
        out_counts_ptr,
        out_p_med_idx_ptr,
        B,
        N,
        G,
        HAS_MASK: tl.constexpr,
        PRE_TMED: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Per-(b, g) segment median and MAD over contiguous sorted slices."""
        b = tl.program_id(0)
        g = tl.program_id(1)
        bg = b * G + g

        seg_start = tl.load(starts_ptr + bg)
        seg_end = tl.load(ends_ptr + bg)
        raw_count = seg_end - seg_start

        if raw_count <= 0:
            tl.store(out_p_med_ptr + bg, 0.0)
            if HAS_MASK:
                tl.store(out_t_med_ptr + bg, 0.0)
            tl.store(out_p_mad_ptr + bg, 0.0)
            tl.store(out_t_mad_ptr + bg, 0.0)
            tl.store(out_counts_ptr + bg, 0.0)
            tl.store(out_p_med_idx_ptr + bg, N)
            return

        base = b * N
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        valid_count = tl.zeros((), dtype=tl.int32)
        if HAS_MASK:
            for start in range(seg_start, seg_end, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_seg = offs < seg_end
                m_val = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
                valid_count += tl.sum((in_seg & (m_val >= 0.5)).to(tl.int32), axis=0)
            if valid_count <= 0:
                tl.store(out_p_med_ptr + bg, 0.0)
                tl.store(out_t_med_ptr + bg, 0.0)
                tl.store(out_p_mad_ptr + bg, 0.0)
                tl.store(out_t_mad_ptr + bg, 0.0)
                tl.store(out_counts_ptr + bg, 0.0)
                tl.store(out_p_med_idx_ptr + bg, N)
                return
            k_rem_p = (valid_count - 1) // 2
            k_rem_t = k_rem_p
            # t_sorted is ascending: extract masked t_med in one scan via running valid count
            t_med = tl.zeros((), dtype=tl.float32)
            seen = 0
            found_t = 0
            for start in range(seg_start, seg_end, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_seg = offs < seg_end
                m_val = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
                ok = in_seg & (m_val >= 0.5)
                c_tile = tl.sum(ok.to(tl.int32), axis=0)
                if (found_t == 0) & (seen + c_tile > k_rem_t):
                    cs = tl.cumsum(ok.to(tl.int32), axis=0)
                    hit = ok & (cs == (k_rem_t - seen + 1))
                    tv = tl.load(t_sorted_ptr + base + offs, mask=hit, other=0.0).to(tl.float32)
                    t_med = tl.sum(tv, axis=0)
                    found_t = 1
                seen += c_tile
            tl.store(out_t_med_ptr + bg, t_med)
            c_f = valid_count.to(tl.float32)
        else:
            k_rem_p = (raw_count - 1) // 2
            if PRE_TMED:
                t_med = tl.load(out_t_med_ptr + bg).to(tl.float32)
            else:
                t_med = tl.load(t_sorted_ptr + base + seg_start + k_rem_p).to(tl.float32)
            c_f = raw_count.to(tl.float32)

        prefix_p = tl.zeros((), dtype=tl.uint32)
        mask_prefix = tl.zeros((), dtype=tl.uint32)
        for pass_i in tl.static_range(4):
            shift = 24 - 8 * pass_i
            hist = tl.zeros([256], dtype=tl.int32)
            for start in range(seg_start, seg_end, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_seg = offs < seg_end
                if HAS_MASK:
                    m_ok = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32) >= 0.5
                    in_seg = in_seg & m_ok
                val = tl.load(p_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                match = in_seg if pass_i == 0 else (in_seg & ((key & mask_prefix) == prefix_p))
                byte_val = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
                h512 = tl.histogram(byte_val, 512)
                hist += tl.sum(tl.reshape(h512, [2, 256]) * keep_lo, axis=0)
            csum = tl.cumsum(hist, axis=0)
            bucket = tl.argmax((csum > k_rem_p).to(tl.int32), axis=0)
            k_rem_p -= tl.sum(tl.where(idx_256 == (bucket - 1), csum, 0), axis=0)
            prefix_p = prefix_p | (bucket.to(tl.uint32) << shift)
            mask_prefix = mask_prefix | (tl.full((), 0xFF, tl.uint32) << shift)

        inv_u = prefix_p ^ tl.where(
            (prefix_p >> 31) == 0,
            tl.full((), 0xFFFFFFFF, tl.uint32),
            tl.full((), 0x80000000, tl.uint32),
        )
        p_med = inv_u.to(tl.float32, bitcast=True)

        p_abs = tl.zeros((), dtype=tl.float32)
        t_abs = tl.zeros((), dtype=tl.float32)
        min_idx = seg_end
        for start in range(seg_start, seg_end, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_seg = offs < seg_end
            if HAS_MASK:
                m_ok = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32) >= 0.5
                in_seg = in_seg & m_ok
            p_val = tl.load(p_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
            t_val = tl.load(t_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
            p_abs += tl.sum(tl.where(in_seg, tl.abs(p_val - p_med), 0.0), axis=0)
            t_abs += tl.sum(tl.where(in_seg, tl.abs(t_val - t_med), 0.0), axis=0)

            pc = tl.where(p_val == 0.0, 0.0, p_val)
            up = pc.to(tl.uint32, bitcast=True)
            kp = up ^ tl.where(
                (up >> 31) != 0,
                tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                tl.full([BLOCK_N], 0x80000000, tl.uint32),
            )
            hit = in_seg & (kp == prefix_p)
            min_idx = tl.minimum(min_idx, tl.min(tl.where(hit, offs, seg_end), axis=0))

        tl.store(out_p_med_ptr + bg, p_med)
        tl.store(out_p_mad_ptr + bg, p_abs / c_f)
        tl.store(out_t_mad_ptr + bg, t_abs / c_f)
        tl.store(out_counts_ptr + bg, c_f)
        tl.store(out_p_med_idx_ptr + bg, tl.where(min_idx < seg_end, min_idx, N))

    @triton.jit
    def _sorted_seg_fwd_kernel(
        p_sorted_ptr,
        t_sorted_ptr,
        m_sorted_ptr,
        starts_ptr,
        ends_ptr,
        p_med_ptr,
        t_med_ptr,
        p_mad_ptr,
        t_mad_ptr,
        non_flat_ptr,
        out_loss_ptr,
        out_sum_s_ptr,
        out_sum_sp_ptr,
        out_sum_signp_ptr,
        B,
        N,
        G,
        HAS_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        g = tl.program_id(1)
        bg = b * G + g

        nf = tl.load(non_flat_ptr + bg).to(tl.float32) > 0.5
        seg_start = tl.load(starts_ptr + bg)
        seg_end = tl.load(ends_ptr + bg)
        if (not nf) | (seg_end <= seg_start):
            tl.store(out_loss_ptr + bg, 0.0)
            tl.store(out_sum_s_ptr + bg, 0.0)
            tl.store(out_sum_sp_ptr + bg, 0.0)
            tl.store(out_sum_signp_ptr + bg, 0.0)
            return

        base = b * N
        pm = tl.load(p_med_ptr + bg).to(tl.float32)
        tm = tl.load(t_med_ptr + bg).to(tl.float32)
        ps = tl.maximum(tl.load(p_mad_ptr + bg).to(tl.float32), 1e-6)
        ts = tl.maximum(tl.load(t_mad_ptr + bg).to(tl.float32), 1e-6)

        l_acc = tl.zeros((), dtype=tl.float32)
        s_acc = tl.zeros((), dtype=tl.float32)
        sp_acc = tl.zeros((), dtype=tl.float32)
        sgn_acc = tl.zeros((), dtype=tl.float32)

        for start in range(seg_start, seg_end, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_seg = offs < seg_end
            if HAS_MASK:
                m_ok = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32) >= 0.5
                in_seg = in_seg & m_ok

            pv = tl.load(p_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
            tv = tl.load(t_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
            dp = pv - pm
            p_norm = dp / ps
            t_norm = (tv - tm) / ts
            err = p_norm - t_norm

            s_val = tl.where(err > 0.0, 1.0, tl.where(err < 0.0, -1.0, 0.0))
            sign_dp = tl.where(dp > 0.0, 1.0, tl.where(dp < 0.0, -1.0, 0.0))

            l_acc += tl.sum(tl.where(in_seg, tl.abs(err), 0.0), axis=0)
            s_acc += tl.sum(tl.where(in_seg, s_val, 0.0), axis=0)
            sp_acc += tl.sum(tl.where(in_seg, s_val * p_norm, 0.0), axis=0)
            sgn_acc += tl.sum(tl.where(in_seg, sign_dp, 0.0), axis=0)

        tl.store(out_loss_ptr + bg, l_acc)
        tl.store(out_sum_s_ptr + bg, s_acc)
        tl.store(out_sum_sp_ptr + bg, sp_acc)
        tl.store(out_sum_signp_ptr + bg, sgn_acc)

    @triton.jit
    def _sorted_seg_bwd_kernel(
        p_sorted_ptr,
        t_sorted_ptr,
        m_sorted_ptr,
        grad_p_sorted_ptr,
        starts_ptr,
        ends_ptr,
        p_med_ptr,
        t_med_ptr,
        p_mad_ptr,
        t_mad_ptr,
        mad_clamped_ptr,
        non_flat_ptr,
        counts_ptr,
        sum_sp_ptr,
        grad_out_ptr,
        denom_ptr,
        B,
        N,
        G_start,
        G_end,
        G_total,
        HAS_MASK: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        g_rel = tl.program_id(1)
        g = G_start + g_rel
        bg = b * G_total + g

        seg_start = tl.load(starts_ptr + bg)
        seg_end = tl.load(ends_ptr + bg)
        if seg_end <= seg_start:
            return

        nf = tl.load(non_flat_ptr + bg).to(tl.float32) > 0.5
        base = b * N

        if not nf:
            if not ACCUMULATE:
                for start in range(seg_start, seg_end, BLOCK_N):
                    offs = start + tl.arange(0, BLOCK_N)
                    tl.store(grad_p_sorted_ptr + base + offs, 0.0, mask=offs < seg_end)
            return

        scale = tl.load(grad_out_ptr).to(tl.float32) / tl.load(denom_ptr).to(tl.float32)
        pm = tl.load(p_med_ptr + bg).to(tl.float32)
        tm = tl.load(t_med_ptr + bg).to(tl.float32)
        ps = tl.maximum(tl.load(p_mad_ptr + bg).to(tl.float32), 1e-6)
        ts = tl.maximum(tl.load(t_mad_ptr + bg).to(tl.float32), 1e-6)
        not_clamped = tl.load(mad_clamped_ptr + bg).to(tl.float32) == 0.0
        cnt = tl.maximum(tl.load(counts_ptr + bg).to(tl.float32), 1.0)
        sum_sp = tl.load(sum_sp_ptr + bg).to(tl.float32)
        mad_coeff = (sum_sp / cnt) if not_clamped else 0.0

        for start in range(seg_start, seg_end, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_seg = offs < seg_end
            if HAS_MASK:
                m_ok = tl.load(m_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32) >= 0.5
                active = in_seg & m_ok
            else:
                active = in_seg

            pv = tl.load(p_sorted_ptr + base + offs, mask=active, other=0.0).to(tl.float32)
            tv = tl.load(t_sorted_ptr + base + offs, mask=active, other=0.0).to(tl.float32)
            dp = pv - pm
            p_norm = dp / ps
            t_norm = (tv - tm) / ts
            err = p_norm - t_norm

            s_val = tl.where(err > 0.0, 1.0, tl.where(err < 0.0, -1.0, 0.0))
            sign_dp = tl.where(dp > 0.0, 1.0, tl.where(dp < 0.0, -1.0, 0.0))
            gp = scale * tl.where(active, (s_val - mad_coeff * sign_dp) / ps, 0.0)

            if ACCUMULATE:
                prev_gp = tl.load(grad_p_sorted_ptr + base + offs, mask=in_seg, other=0.0).to(tl.float32)
                gp = prev_gp + gp
            tl.store(grad_p_sorted_ptr + base + offs, gp, mask=in_seg)

    @triton.jit
    def _grad_norm_fwd_kernel(
        p_ptr,
        t_ptr,
        m_ptr,
        out_grad_ptr,
        out_norm_ptr,
        out_dm_grad_ptr,
        out_dm_norm_ptr,
        B,
        H,
        W,
        stride_b,
        stride_h,
        stride_w,
        normal_beta,
        HAS_MASK: tl.constexpr,
        COMPUTE_NORMAL: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(W, BLOCK_W)
        b = pid // (H * num_x_blocks)
        rem = pid % (H * num_x_blocks)
        y = rem // num_x_blocks
        xb = rem % num_x_blocks

        offs_x = xb * BLOCK_W + tl.arange(0, BLOCK_W)
        in_w = offs_x < W
        base = b * stride_b + y * stride_h

        p0 = tl.load(p_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)
        t0 = tl.load(t_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)

        in_x1 = (offs_x + 1) < W
        p_r = tl.load(p_ptr + base + (offs_x + 1) * stride_w, mask=in_x1, other=0.0).to(tl.float32)
        t_r = tl.load(t_ptr + base + (offs_x + 1) * stride_w, mask=in_x1, other=0.0).to(tl.float32)

        in_y1 = (y + 1) < H
        in_b = in_w & in_y1
        p_d = tl.load(p_ptr + base + stride_h + offs_x * stride_w, mask=in_b, other=0.0).to(tl.float32)
        t_d = tl.load(t_ptr + base + stride_h + offs_x * stride_w, mask=in_b, other=0.0).to(tl.float32)

        p_dx = p_r - p0
        t_dx = t_r - t0
        p_dy = p_d - p0
        t_dy = t_d - t0

        if HAS_MASK:
            m0 = tl.load(m_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)
            m_r = tl.load(m_ptr + base + (offs_x + 1) * stride_w, mask=in_x1, other=0.0).to(tl.float32)
            m_d = tl.load(m_ptr + base + stride_h + offs_x * stride_w, mask=in_b, other=0.0).to(tl.float32)
            mx = tl.where(in_x1, m0 * m_r, 0.0)
            my = tl.where(in_b, m0 * m_d, 0.0)
            tl.store(out_dm_grad_ptr + pid, tl.sum(mx + my, axis=0))
        else:
            mx = tl.where(in_x1, 1.0, 0.0)
            my = tl.where(in_b, 1.0, 0.0)

        err_x = tl.abs(p_dx - t_dx) * mx
        err_y = tl.abs(p_dy - t_dy) * my
        tl.store(out_grad_ptr + pid, tl.sum(err_x + err_y, axis=0))

        if COMPUTE_NORMAL:
            valid_xy = in_x1 & in_y1
            px = normal_beta * p_dx
            py = normal_beta * p_dy
            tx = normal_beta * t_dx
            ty = normal_beta * t_dy
            rp = tl.rsqrt(px * px + py * py + 1.0)
            rt = tl.rsqrt(tx * tx + ty * ty + 1.0)
            cos_sim = (px * tx + py * ty + 1.0) * rp * rt
            if HAS_MASK:
                mxy = tl.where(valid_xy, mx * my, 0.0)
                tl.store(out_dm_norm_ptr + pid, tl.sum(mxy, axis=0))
            else:
                mxy = tl.where(valid_xy, 1.0, 0.0)
            tl.store(out_norm_ptr + pid, tl.sum((1.0 - cos_sim) * mxy, axis=0))

    @triton.jit
    def _pool2x_pair_kernel(
        p_in_ptr,
        t_in_ptr,
        m_in_ptr,
        p_out_ptr,
        t_out_ptr,
        m_out_ptr,
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        HAS_MASK: tl.constexpr,
        BLOCK_X: tl.constexpr,
    ):
        """Downsample (p, t[, m]) by 2x2 average in one pass matching F.avg_pool2d(..., 2)."""
        pid = tl.program_id(0)
        num_xb = tl.cdiv(W_out, BLOCK_X)
        b = pid // (H_out * num_xb)
        rem = pid % (H_out * num_xb)
        y = rem // num_xb
        xb = rem % num_xb

        ox = xb * BLOCK_X + tl.arange(0, BLOCK_X)
        in_w = ox < W_out
        base_in0 = (b * H_in + 2 * y) * W_in
        base_in1 = base_in0 + W_in
        x0 = 2 * ox
        x1 = x0 + 1

        p00 = tl.load(p_in_ptr + base_in0 + x0, mask=in_w, other=0.0).to(tl.float32)
        p01 = tl.load(p_in_ptr + base_in0 + x1, mask=in_w, other=0.0).to(tl.float32)
        p10 = tl.load(p_in_ptr + base_in1 + x0, mask=in_w, other=0.0).to(tl.float32)
        p11 = tl.load(p_in_ptr + base_in1 + x1, mask=in_w, other=0.0).to(tl.float32)

        t00 = tl.load(t_in_ptr + base_in0 + x0, mask=in_w, other=0.0).to(tl.float32)
        t01 = tl.load(t_in_ptr + base_in0 + x1, mask=in_w, other=0.0).to(tl.float32)
        t10 = tl.load(t_in_ptr + base_in1 + x0, mask=in_w, other=0.0).to(tl.float32)
        t11 = tl.load(t_in_ptr + base_in1 + x1, mask=in_w, other=0.0).to(tl.float32)

        out_off = (b * H_out + y) * W_out + ox
        tl.store(p_out_ptr + out_off, 0.25 * (p00 + p01 + p10 + p11), mask=in_w)
        tl.store(t_out_ptr + out_off, 0.25 * (t00 + t01 + t10 + t11), mask=in_w)

        if HAS_MASK:
            m00 = tl.load(m_in_ptr + base_in0 + x0, mask=in_w, other=0.0).to(tl.float32)
            m01 = tl.load(m_in_ptr + base_in0 + x1, mask=in_w, other=0.0).to(tl.float32)
            m10 = tl.load(m_in_ptr + base_in1 + x0, mask=in_w, other=0.0).to(tl.float32)
            m11 = tl.load(m_in_ptr + base_in1 + x1, mask=in_w, other=0.0).to(tl.float32)
            m_avg = 0.25 * (m00 + m01 + m10 + m11)
            tl.store(m_out_ptr + out_off, tl.where(m_avg > 0.99, 1.0, 0.0), mask=in_w)

    @triton.jit
    def _grad_norm_bwd_kernel(
        p_ptr,
        t_ptr,
        m_ptr,
        up_grad_ptr,
        grad_p_ptr,
        w_grad_ptr,
        w_norm_ptr,
        B,
        H,
        W,
        H_up,
        W_up,
        stride_b,
        stride_h,
        stride_w,
        normal_beta,
        HAS_MASK: tl.constexpr,
        COMPUTE_NORMAL: tl.constexpr,
        ADD_UP: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(W, BLOCK_W)
        b = pid // (H * num_x_blocks)
        rem = pid % (H * num_x_blocks)
        y = rem // num_x_blocks
        xb = rem % num_x_blocks

        w_grad = tl.load(w_grad_ptr).to(tl.float32)
        w_norm = tl.load(w_norm_ptr).to(tl.float32) if COMPUTE_NORMAL else 0.0

        offs_x = xb * BLOCK_W + tl.arange(0, BLOCK_W)
        in_w = offs_x < W
        base = b * stride_b + y * stride_h

        p0 = tl.load(p_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)
        t0 = tl.load(t_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)

        in_r = (offs_x + 1) < W
        in_l = in_w & (offs_x > 0)
        in_d = in_w & ((y + 1) < H)
        in_u = in_w & (y > 0)

        p_r = tl.load(p_ptr + base + (offs_x + 1) * stride_w, mask=in_r, other=0.0).to(tl.float32)
        t_r = tl.load(t_ptr + base + (offs_x + 1) * stride_w, mask=in_r, other=0.0).to(tl.float32)
        p_d = tl.load(p_ptr + base + stride_h + offs_x * stride_w, mask=in_d, other=0.0).to(tl.float32)
        t_d = tl.load(t_ptr + base + stride_h + offs_x * stride_w, mask=in_d, other=0.0).to(tl.float32)
        p_l = tl.load(p_ptr + base + (offs_x - 1) * stride_w, mask=in_l, other=0.0).to(tl.float32)
        t_l = tl.load(t_ptr + base + (offs_x - 1) * stride_w, mask=in_l, other=0.0).to(tl.float32)
        p_u = tl.load(p_ptr + base - stride_h + offs_x * stride_w, mask=in_u, other=0.0).to(tl.float32)
        t_u = tl.load(t_ptr + base - stride_h + offs_x * stride_w, mask=in_u, other=0.0).to(tl.float32)

        if HAS_MASK:
            m0 = tl.load(m_ptr + base + offs_x * stride_w, mask=in_w, other=0.0).to(tl.float32)
            m_r = tl.load(m_ptr + base + (offs_x + 1) * stride_w, mask=in_r, other=0.0).to(tl.float32)
            m_d = tl.load(m_ptr + base + stride_h + offs_x * stride_w, mask=in_d, other=0.0).to(tl.float32)
            m_l = tl.load(m_ptr + base + (offs_x - 1) * stride_w, mask=in_l, other=0.0).to(tl.float32)
            m_u = tl.load(m_ptr + base - stride_h + offs_x * stride_w, mask=in_u, other=0.0).to(tl.float32)
            mx_c = tl.where(in_r, m0 * m_r, 0.0)
            my_c = tl.where(in_d, m0 * m_d, 0.0)
            mx_l = tl.where(in_l, m_l * m0, 0.0)
            my_u = tl.where(in_u, m_u * m0, 0.0)
        else:
            mx_c = tl.where(in_r, 1.0, 0.0)
            my_c = tl.where(in_d, 1.0, 0.0)
            mx_l = tl.where(in_l, 1.0, 0.0)
            my_u = tl.where(in_u, 1.0, 0.0)

        diff_xc = (p_r - p0) - (t_r - t0)
        s_xc = tl.where(diff_xc > 0.0, 1.0, tl.where(diff_xc < 0.0, -1.0, 0.0)) * w_grad * mx_c
        diff_xl = (p0 - p_l) - (t0 - t_l)
        s_xl = tl.where(diff_xl > 0.0, 1.0, tl.where(diff_xl < 0.0, -1.0, 0.0)) * w_grad * mx_l
        diff_yc = (p_d - p0) - (t_d - t0)
        s_yc = tl.where(diff_yc > 0.0, 1.0, tl.where(diff_yc < 0.0, -1.0, 0.0)) * w_grad * my_c
        diff_yu = (p0 - p_u) - (t0 - t_u)
        s_yu = tl.where(diff_yu > 0.0, 1.0, tl.where(diff_yu < 0.0, -1.0, 0.0)) * w_grad * my_u

        gp = s_xl - s_xc + s_yu - s_yc

        if COMPUTE_NORMAL:
            valid_c = in_r & in_d
            px_c = normal_beta * (p_r - p0)
            py_c = normal_beta * (p_d - p0)
            tx_c = normal_beta * (t_r - t0)
            ty_c = normal_beta * (t_d - t0)
            rp_c = tl.rsqrt(px_c * px_c + py_c * py_c + 1.0)
            rt_c = tl.rsqrt(tx_c * tx_c + ty_c * ty_c + 1.0)
            cos_c = (px_c * tx_c + py_c * ty_c + 1.0) * rp_c * rt_c
            mxy_c = tl.where(valid_c, mx_c * my_c, 0.0) if HAS_MASK else tl.where(valid_c, 1.0, 0.0)
            w_c = w_norm * normal_beta * mxy_c
            dpx_c = w_c * rp_c * (cos_c * px_c * rp_c - tx_c * rt_c)
            dpy_c = w_c * rp_c * (cos_c * py_c * rp_c - ty_c * rt_c)
            gp += -(dpx_c + dpy_c)

            valid_l = in_l & in_d
            p_ld = tl.load(p_ptr + base + stride_h + (offs_x - 1) * stride_w, mask=valid_l, other=0.0).to(tl.float32)
            t_ld = tl.load(t_ptr + base + stride_h + (offs_x - 1) * stride_w, mask=valid_l, other=0.0).to(tl.float32)
            px_l = normal_beta * (p0 - p_l)
            py_l = normal_beta * (p_ld - p_l)
            tx_l = normal_beta * (t0 - t_l)
            ty_l = normal_beta * (t_ld - t_l)
            rp_l = tl.rsqrt(px_l * px_l + py_l * py_l + 1.0)
            rt_l = tl.rsqrt(tx_l * tx_l + ty_l * ty_l + 1.0)
            cos_l = (px_l * tx_l + py_l * ty_l + 1.0) * rp_l * rt_l
            if HAS_MASK:
                m_ld = tl.load(m_ptr + base + stride_h + (offs_x - 1) * stride_w, mask=valid_l, other=0.0).to(tl.float32)
                mxy_l = tl.where(valid_l, (m_l * m0) * (m_l * m_ld), 0.0)
            else:
                mxy_l = tl.where(valid_l, 1.0, 0.0)
            dpx_l = (w_norm * normal_beta * mxy_l) * rp_l * (cos_l * px_l * rp_l - tx_l * rt_l)
            gp += dpx_l

            valid_u = in_u & in_r
            p_ur = tl.load(p_ptr + base - stride_h + (offs_x + 1) * stride_w, mask=valid_u, other=0.0).to(tl.float32)
            t_ur = tl.load(t_ptr + base - stride_h + (offs_x + 1) * stride_w, mask=valid_u, other=0.0).to(tl.float32)
            px_u = normal_beta * (p_ur - p_u)
            py_u = normal_beta * (p0 - p_u)
            tx_u = normal_beta * (t_ur - t_u)
            ty_u = normal_beta * (t0 - t_u)
            rp_u = tl.rsqrt(px_u * px_u + py_u * py_u + 1.0)
            rt_u = tl.rsqrt(tx_u * tx_u + ty_u * ty_u + 1.0)
            cos_u = (px_u * tx_u + py_u * ty_u + 1.0) * rp_u * rt_u
            if HAS_MASK:
                m_ur = tl.load(m_ptr + base - stride_h + (offs_x + 1) * stride_w, mask=valid_u, other=0.0).to(tl.float32)
                mxy_u = tl.where(valid_u, (m_u * m_ur) * (m_u * m0), 0.0)
            else:
                mxy_u = tl.where(valid_u, 1.0, 0.0)
            dpy_u = (w_norm * normal_beta * mxy_u) * rp_u * (cos_u * py_u * rp_u - ty_u * rt_u)
            gp += dpy_u

        if ADD_UP:
            y_up = y // 2
            x_up = offs_x // 2
            in_up = in_w & (y_up < H_up) & (x_up < W_up)
            gp += 0.25 * tl.load(
                up_grad_ptr + (b * H_up + y_up) * W_up + x_up,
                mask=in_up,
                other=0.0,
            ).to(tl.float32)

        tl.store(grad_p_ptr + base + offs_x * stride_w, gp, mask=in_w)

    @triton.jit
    def _q3_packed1024_kernel(
        x_ptr,
        out_q_ptr,
        B,
        N,
        k0,
        k1,
        k2,
        BLOCK_N: tl.constexpr,
    ):
        """Find exact (q25, q50, q75) lower-rank order statistics per batch row without sorting."""
        b = tl.program_id(0)
        base = b * N
        kr0 = k0
        kr1 = k1
        kr2 = k2
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        h_init = tl.zeros([256], dtype=tl.int32)
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_b = offs < N
            val = tl.load(x_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            val = tl.where(val == 0.0, 0.0, val)
            u = val.to(tl.uint32, bitcast=True)
            key = u ^ tl.where(
                (u >> 31) != 0,
                tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                tl.full([BLOCK_N], 0x80000000, tl.uint32),
            )
            bv = tl.where(in_b, ((key >> 24) & 0xFF).to(tl.int32), 256)
            h_init += tl.sum(tl.reshape(tl.histogram(bv, 512), [2, 256]) * keep_lo, axis=0)

        c_i = tl.cumsum(h_init, axis=0)
        bk0 = tl.argmax((c_i > kr0).to(tl.int32), axis=0)
        kr0 -= tl.sum(tl.where(idx_256 == (bk0 - 1), c_i, 0), axis=0)
        p0 = bk0.to(tl.uint32) << 24
        bk1 = tl.argmax((c_i > kr1).to(tl.int32), axis=0)
        kr1 -= tl.sum(tl.where(idx_256 == (bk1 - 1), c_i, 0), axis=0)
        p1 = bk1.to(tl.uint32) << 24
        bk2 = tl.argmax((c_i > kr2).to(tl.int32), axis=0)
        kr2 -= tl.sum(tl.where(idx_256 == (bk2 - 1), c_i, 0), axis=0)
        p2 = bk2.to(tl.uint32) << 24
        mp = tl.full((), 0xFF000000, tl.uint32)

        rows_4 = tl.arange(0, 4)[:, None]
        for pass_i in tl.static_range(1, 4):
            shift = 24 - 8 * pass_i
            h0 = tl.zeros([256], dtype=tl.int32)
            h1 = tl.zeros([256], dtype=tl.int32)
            h2 = tl.zeros([256], dtype=tl.int32)
            same01 = p1 == p0
            same02 = p2 == p0
            same12 = p2 == p1
            for start in range(0, N, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_b = offs < N
                val = tl.load(x_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                km = key & mp
                bs = ((key >> shift) & 0xFF).to(tl.int32)
                slot = tl.where(km == p0, 0, tl.where(km == p1, 256, tl.where(km == p2, 512, 768)))
                bval = tl.where(in_b, slot + bs, 768)
                hr = tl.reshape(tl.histogram(bval, 1024), [4, 256])
                h0 += tl.sum(tl.where(rows_4 == 0, hr, 0), axis=0)
                h1 += tl.sum(tl.where(rows_4 == 1, hr, 0), axis=0)
                h2 += tl.sum(tl.where(rows_4 == 2, hr, 0), axis=0)

            if same01:
                h1 = h0
            if same02:
                h2 = h0
            elif same12:
                h2 = h1

            c0 = tl.cumsum(h0, axis=0)
            bk0 = tl.argmax((c0 > kr0).to(tl.int32), axis=0)
            kr0 -= tl.sum(tl.where(idx_256 == (bk0 - 1), c0, 0), axis=0)
            p0 |= bk0.to(tl.uint32) << shift

            c1 = tl.cumsum(h1, axis=0)
            bk1 = tl.argmax((c1 > kr1).to(tl.int32), axis=0)
            kr1 -= tl.sum(tl.where(idx_256 == (bk1 - 1), c1, 0), axis=0)
            p1 |= bk1.to(tl.uint32) << shift

            c2 = tl.cumsum(h2, axis=0)
            bk2 = tl.argmax((c2 > kr2).to(tl.int32), axis=0)
            kr2 -= tl.sum(tl.where(idx_256 == (bk2 - 1), c2, 0), axis=0)
            p2 |= bk2.to(tl.uint32) << shift

            mp |= tl.full((), 0xFF, tl.uint32) << shift

        inv0 = p0 ^ tl.where((p0 >> 31) == 0, tl.full((), 0xFFFFFFFF, tl.uint32), tl.full((), 0x80000000, tl.uint32))
        inv1 = p1 ^ tl.where((p1 >> 31) == 0, tl.full((), 0xFFFFFFFF, tl.uint32), tl.full((), 0x80000000, tl.uint32))
        inv2 = p2 ^ tl.where((p2 >> 31) == 0, tl.full((), 0xFFFFFFFF, tl.uint32), tl.full((), 0x80000000, tl.uint32))
        tl.store(out_q_ptr + b * 3 + 0, inv0.to(tl.float32, bitcast=True))
        tl.store(out_q_ptr + b * 3 + 1, inv1.to(tl.float32, bitcast=True))
        tl.store(out_q_ptr + b * 3 + 2, inv2.to(tl.float32, bitcast=True))

    @triton.jit
    def _dr_bounds_kernel(
        x_ptr,
        out_q_ptr,
        B,
        N,
        BLOCK_N: tl.constexpr,
    ):
        """Compute uniform min/max 4-bin split thresholds [q0, q1, q2] for HDN-DR mode."""
        b = tl.program_id(0)
        base = b * N
        tmin = tl.full((), 1e38, dtype=tl.float32)
        tmax = tl.full((), -1e38, dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_b = offs < N
            v_lo = tl.load(x_ptr + base + offs, mask=in_b, other=1e38).to(tl.float32)
            v_hi = tl.where(in_b, v_lo, -1e38)
            tmin = tl.minimum(tmin, tl.min(v_lo, axis=0))
            tmax = tl.maximum(tmax, tl.max(v_hi, axis=0))
        span = tl.maximum(tmax - tmin, 1e-6)
        tl.store(out_q_ptr + b * 3 + 0, tmin + 0.25 * span)
        tl.store(out_q_ptr + b * 3 + 1, tmin + 0.50 * span)
        tl.store(out_q_ptr + b * 3 + 2, tmin + 0.75 * span)

    @triton.jit
    def _bucket4_count_kernel(
        t_ptr,
        q_ptr,
        tile_cnt_ptr,
        seg_starts_ptr,
        seg_ends_ptr,
        B,
        N,
        num_tiles,
        STRICT_LT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        base = b * N
        q0 = tl.load(q_ptr + b * 3 + 0).to(tl.float32)
        q1 = tl.load(q_ptr + b * 3 + 1).to(tl.float32)
        q2 = tl.load(q_ptr + b * 3 + 2).to(tl.float32)
        c0 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        for tile in range(num_tiles):
            offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            in_b = offs < N
            tv = tl.load(t_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
            if STRICT_LT:
                b0 = in_b & (tv < q0)
                b1 = in_b & (tv >= q0) & (tv < q1)
                b2 = in_b & (tv >= q1) & (tv < q2)
                b3 = in_b & (tv >= q2)
            else:
                b0 = in_b & (tv <= q0)
                b1 = in_b & (tv > q0) & (tv <= q1)
                b2 = in_b & (tv > q1) & (tv <= q2)
                b3 = in_b & (tv > q2)
            tb = (b * num_tiles + tile) * 4
            tl.store(tile_cnt_ptr + tb + 0, c0)
            tl.store(tile_cnt_ptr + tb + 1, c1)
            tl.store(tile_cnt_ptr + tb + 2, c2)
            tl.store(tile_cnt_ptr + tb + 3, c3)
            c0 += tl.sum(b0.to(tl.int32), axis=0)
            c1 += tl.sum(b1.to(tl.int32), axis=0)
            c2 += tl.sum(b2.to(tl.int32), axis=0)
            c3 += tl.sum(b3.to(tl.int32), axis=0)
        o1 = c0
        o2 = o1 + c1
        o3 = o2 + c2
        sb = b * 6
        tl.store(seg_starts_ptr + sb + 0, 0)
        tl.store(seg_ends_ptr + sb + 0, o2)
        tl.store(seg_starts_ptr + sb + 1, o2)
        tl.store(seg_ends_ptr + sb + 1, N)
        tl.store(seg_starts_ptr + sb + 2, 0)
        tl.store(seg_ends_ptr + sb + 2, o1)
        tl.store(seg_starts_ptr + sb + 3, o1)
        tl.store(seg_ends_ptr + sb + 3, o2)
        tl.store(seg_starts_ptr + sb + 4, o2)
        tl.store(seg_ends_ptr + sb + 4, o3)
        tl.store(seg_starts_ptr + sb + 5, o3)
        tl.store(seg_ends_ptr + sb + 5, N)

    @triton.jit
    def _bucket4_scatter_kernel(
        p_ptr,
        t_ptr,
        q_ptr,
        tile_cnt_ptr,
        seg_starts_ptr,
        out_p_ptr,
        out_t_ptr,
        out_idx_ptr,
        B,
        N,
        num_tiles,
        STRICT_LT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        base = b * N
        offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_b = offs < N

        q0 = tl.load(q_ptr + b * 3 + 0).to(tl.float32)
        q1 = tl.load(q_ptr + b * 3 + 1).to(tl.float32)
        q2 = tl.load(q_ptr + b * 3 + 2).to(tl.float32)

        tv = tl.load(t_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)
        pv = tl.load(p_ptr + base + offs, mask=in_b, other=0.0).to(tl.float32)

        if STRICT_LT:
            b0 = in_b & (tv < q0)
            b1 = in_b & (tv >= q0) & (tv < q1)
            b2 = in_b & (tv >= q1) & (tv < q2)
            b3 = in_b & (tv >= q2)
        else:
            b0 = in_b & (tv <= q0)
            b1 = in_b & (tv > q0) & (tv <= q1)
            b2 = in_b & (tv > q1) & (tv <= q2)
            b3 = in_b & (tv > q2)

        tb = (b * num_tiles + tile) * 4
        sb = b * 6
        off0 = tl.load(seg_starts_ptr + sb + 2) + tl.load(tile_cnt_ptr + tb + 0)
        off1 = tl.load(seg_starts_ptr + sb + 3) + tl.load(tile_cnt_ptr + tb + 1)
        off2 = tl.load(seg_starts_ptr + sb + 4) + tl.load(tile_cnt_ptr + tb + 2)
        off3 = tl.load(seg_starts_ptr + sb + 5) + tl.load(tile_cnt_ptr + tb + 3)

        pos0 = off0 + tl.cumsum(b0.to(tl.int32), axis=0) - 1
        pos1 = off1 + tl.cumsum(b1.to(tl.int32), axis=0) - 1
        pos2 = off2 + tl.cumsum(b2.to(tl.int32), axis=0) - 1
        pos3 = off3 + tl.cumsum(b3.to(tl.int32), axis=0) - 1

        dst = tl.where(b0, pos0, tl.where(b1, pos1, tl.where(b2, pos2, pos3)))
        tl.store(out_p_ptr + base + dst, pv.to(out_p_ptr.dtype.element_ty), mask=in_b)
        tl.store(out_t_ptr + base + dst, tv, mask=in_b)
        tl.store(out_idx_ptr + base + dst, offs, mask=in_b)

    @triton.jit
    def _seg_t_med_kernel(
        t_part_ptr,
        starts_ptr,
        ends_ptr,
        out_t_med_ptr,
        B,
        N,
        G,
        BLOCK_N: tl.constexpr,
    ):
        """Exact segment median on partitioned (unsorted within bucket) target slices."""
        b = tl.program_id(0)
        g = tl.program_id(1)
        bg = b * G + g
        seg_start = tl.load(starts_ptr + bg)
        seg_end = tl.load(ends_ptr + bg)
        cnt = seg_end - seg_start
        if cnt <= 0:
            tl.store(out_t_med_ptr + bg, 0.0)
            return

        base = b * N
        k_rem = (cnt - 1) // 2
        prefix = tl.zeros((), dtype=tl.uint32)
        mp = tl.zeros((), dtype=tl.uint32)
        idx_256 = tl.arange(0, 256)
        keep_lo = (1 - tl.arange(0, 2))[:, None]

        for pass_i in tl.static_range(4):
            shift = 24 - 8 * pass_i
            hist = tl.zeros([256], dtype=tl.int32)
            for start in range(seg_start, seg_end, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_s = offs < seg_end
                val = tl.load(t_part_ptr + base + offs, mask=in_s, other=0.0).to(tl.float32)
                val = tl.where(val == 0.0, 0.0, val)
                u = val.to(tl.uint32, bitcast=True)
                key = u ^ tl.where(
                    (u >> 31) != 0,
                    tl.full([BLOCK_N], 0xFFFFFFFF, tl.uint32),
                    tl.full([BLOCK_N], 0x80000000, tl.uint32),
                )
                match = in_s if pass_i == 0 else (in_s & ((key & mp) == prefix))
                bv = tl.where(match, ((key >> shift) & 0xFF).to(tl.int32), 256)
                hist += tl.sum(tl.reshape(tl.histogram(bv, 512), [2, 256]) * keep_lo, axis=0)
            csum = tl.cumsum(hist, axis=0)
            bk = tl.argmax((csum > k_rem).to(tl.int32), axis=0)
            k_rem -= tl.sum(tl.where(idx_256 == (bk - 1), csum, 0), axis=0)
            prefix |= bk.to(tl.uint32) << shift
            mp |= tl.full((), 0xFF, tl.uint32) << shift

        inv_u = prefix ^ tl.where((prefix >> 31) == 0, tl.full((), 0xFFFFFFFF, tl.uint32), tl.full((), 0x80000000, tl.uint32))
        tl.store(out_t_med_ptr + bg, inv_u.to(tl.float32, bitcast=True))

    @triton.jit
    def _bucket_bwd_scatter_kernel(
        p_part_ptr,
        t_part_ptr,
        orig_idx_ptr,
        grad_p_ptr,
        starts_ptr,
        ends_ptr,
        p_med_ptr,
        t_med_ptr,
        p_mad_ptr,
        t_mad_ptr,
        non_flat_ptr,
        counts_ptr,
        p_med_idx_ptr,
        sum_s_ptr,
        sum_sp_ptr,
        sum_signp_ptr,
        grad_out_ptr,
        denom_ptr,
        B,
        N,
        BLOCK_N: tl.constexpr,
    ):
        """Fused S=2 + S=4 backward with in-register median scatter and direct pred-dtype output."""
        b = tl.program_id(0)
        k = tl.program_id(1)
        g4 = 2 + k
        g2 = k // 2
        bg4 = b * 6 + g4
        bg2 = b * 6 + g2

        seg_start = tl.load(starts_ptr + bg4)
        seg_end = tl.load(ends_ptr + bg4)
        if seg_end <= seg_start:
            return

        nf4 = tl.load(non_flat_ptr + bg4).to(tl.float32) > 0.5
        nf2 = tl.load(non_flat_ptr + bg2).to(tl.float32) > 0.5
        base = b * N

        if (not nf4) & (not nf2):
            zero_vec = tl.zeros([BLOCK_N], dtype=grad_p_ptr.dtype.element_ty)
            for start in range(seg_start, seg_end, BLOCK_N):
                offs = start + tl.arange(0, BLOCK_N)
                in_s = offs < seg_end
                oidx = tl.load(orig_idx_ptr + base + offs, mask=in_s, other=0)
                tl.store(grad_p_ptr + base + oidx, zero_vec, mask=in_s)
            return

        scale = tl.load(grad_out_ptr).to(tl.float32) / tl.load(denom_ptr).to(tl.float32)

        pm4 = tl.load(p_med_ptr + bg4).to(tl.float32)
        tm4 = tl.load(t_med_ptr + bg4).to(tl.float32)
        pmad4 = tl.load(p_mad_ptr + bg4).to(tl.float32)
        ps4 = tl.maximum(pmad4, 1e-6)
        ts4 = tl.maximum(tl.load(t_mad_ptr + bg4).to(tl.float32), 1e-6)
        cnt4 = tl.maximum(tl.load(counts_ptr + bg4).to(tl.float32), 1.0)
        ssp4 = tl.load(sum_sp_ptr + bg4).to(tl.float32)
        ss4 = tl.load(sum_s_ptr + bg4).to(tl.float32)
        sgnp4 = tl.load(sum_signp_ptr + bg4).to(tl.float32)
        mc4 = (ssp4 / cnt4) if (pmad4 >= 1e-6) else 0.0
        gmed4 = (scale * (-ss4 + mc4 * sgnp4) / ps4) if nf4 else 0.0
        mid4 = tl.load(p_med_idx_ptr + bg4)

        pm2 = tl.load(p_med_ptr + bg2).to(tl.float32)
        tm2 = tl.load(t_med_ptr + bg2).to(tl.float32)
        pmad2 = tl.load(p_mad_ptr + bg2).to(tl.float32)
        ps2 = tl.maximum(pmad2, 1e-6)
        ts2 = tl.maximum(tl.load(t_mad_ptr + bg2).to(tl.float32), 1e-6)
        cnt2 = tl.maximum(tl.load(counts_ptr + bg2).to(tl.float32), 1.0)
        ssp2 = tl.load(sum_sp_ptr + bg2).to(tl.float32)
        ss2 = tl.load(sum_s_ptr + bg2).to(tl.float32)
        sgnp2 = tl.load(sum_signp_ptr + bg2).to(tl.float32)
        mc2 = (ssp2 / cnt2) if (pmad2 >= 1e-6) else 0.0
        gmed2 = (scale * (-ss2 + mc2 * sgnp2) / ps2) if nf2 else 0.0
        mid2 = tl.load(p_med_idx_ptr + bg2)

        for start in range(seg_start, seg_end, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_s = offs < seg_end
            pv = tl.load(p_part_ptr + base + offs, mask=in_s, other=0.0).to(tl.float32)
            tv = tl.load(t_part_ptr + base + offs, mask=in_s, other=0.0).to(tl.float32)
            oidx = tl.load(orig_idx_ptr + base + offs, mask=in_s, other=0)

            gp = tl.zeros([BLOCK_N], dtype=tl.float32)
            if nf4:
                dp4 = pv - pm4
                err4 = dp4 / ps4 - (tv - tm4) / ts4
                s4 = tl.where(err4 > 0.0, 1.0, tl.where(err4 < 0.0, -1.0, 0.0))
                sd4 = tl.where(dp4 > 0.0, 1.0, tl.where(dp4 < 0.0, -1.0, 0.0))
                gp += scale * (s4 - mc4 * sd4) / ps4 + tl.where(offs == mid4, gmed4, 0.0)
            if nf2:
                dp2 = pv - pm2
                err2 = dp2 / ps2 - (tv - tm2) / ts2
                s2 = tl.where(err2 > 0.0, 1.0, tl.where(err2 < 0.0, -1.0, 0.0))
                sd2 = tl.where(dp2 > 0.0, 1.0, tl.where(dp2 < 0.0, -1.0, 0.0))
                gp += scale * (s2 - mc2 * sd2) / ps2 + tl.where(offs == mid2, gmed2, 0.0)

            tl.store(grad_p_ptr + base + oidx, gp.to(grad_p_ptr.dtype.element_ty), mask=in_s)


# ---------------------------------------------------------------------------
# Autograd wrappers (Triton CUDA paths)
# ---------------------------------------------------------------------------

class _TritonNormalize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        t_med_pre: Optional[torch.Tensor],
        mad_denom_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W = pred.shape
        N = H * W
        device = pred.device
        p_flat = pred.contiguous().view(B, N)
        t_flat = target.contiguous().view(B, N)
        has_mask = mask is not None
        m_flat = mask.contiguous().view(B, N) if has_mask else p_flat

        block_med = 2048 if N > 512 else 512
        p_med = torch.empty(B, device=device, dtype=torch.float32)
        p_idx = torch.empty(B, device=device, dtype=torch.int64)
        t_med = torch.empty(B, device=device, dtype=torch.float32)

        if has_mask:
            _masked_median_kernel[(B,)](p_flat, m_flat, p_med, p_idx, B, N, FIND_IDX=True, BLOCK_N=block_med)
            _masked_median_kernel[(B,)](t_flat, m_flat, t_med, p_idx, B, N, FIND_IDX=False, BLOCK_N=block_med)
        else:
            _dense_median_kernel[(B,)](p_flat, p_med, p_idx, B, N, FIND_IDX=True, BLOCK_N=block_med)
            if t_med_pre is None:
                _dense_median_kernel[(B,)](t_flat, t_med, p_idx, B, N, FIND_IDX=False, BLOCK_N=block_med)
            else:
                t_med = t_med_pre.contiguous().view(B).to(torch.float32)

        num_tiles_mad = triton.cdiv(N, block_med)
        p_part = torch.empty((B, num_tiles_mad), device=device, dtype=torch.float32)
        t_part = torch.empty((B, num_tiles_mad), device=device, dtype=torch.float32)
        m_part = torch.empty((B, num_tiles_mad), device=device, dtype=torch.float32) if has_mask else p_part

        _norm_mad_tile_kernel[(B, num_tiles_mad)](
            p_flat, t_flat, m_flat, p_med, t_med,
            p_part, t_part, m_part,
            B, N, num_tiles_mad, HAS_MASK=has_mask, BLOCK_N=block_med,
        )
        if has_mask:
            M = (m_part.sum(dim=1) + mad_denom_eps).clamp(min=1e-8)
        else:
            M = torch.full((B,), float(N) + mad_denom_eps, device=device, dtype=torch.float32).clamp(min=1e-8)
        p_mad = p_part.sum(dim=1) / M
        t_mad = t_part.sum(dim=1) / M

        p_hat = torch.empty((B, H, W), device=device, dtype=torch.float32)
        t_hat = torch.empty((B, H, W), device=device, dtype=torch.float32)
        block_apply = 1024 if N > 512 else 512
        _norm_apply_fwd_kernel[(B, triton.cdiv(N, block_apply))](
            p_flat, t_flat, p_med, t_med, p_mad, t_mad,
            p_hat.view(B, N), t_hat.view(B, N), B, N, BLOCK_N=block_apply,
        )

        ctx.save_for_backward(p_flat, m_flat, p_med, p_mad, p_idx, M)
        ctx.has_mask = has_mask
        ctx.block_med = block_med
        ctx.block_apply = block_apply
        ctx.pred_dtype = pred.dtype
        return p_hat, t_hat

    @staticmethod
    def backward(
        ctx: Any,
        d_phat: torch.Tensor,
        d_that: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        p_flat, m_flat, p_med, p_mad, p_idx, M = ctx.saved_tensors
        B, N = p_flat.shape
        device = p_flat.device
        dp_flat = d_phat.contiguous().view(B, N)

        num_tiles = triton.cdiv(N, ctx.block_med)
        g1_p = torch.empty((B, num_tiles), device=device, dtype=torch.float32)
        g2_p = torch.empty((B, num_tiles), device=device, dtype=torch.float32)
        gs_p = torch.empty((B, num_tiles), device=device, dtype=torch.float32)
        _norm_bwd_tile_kernel[(B, num_tiles)](
            p_flat, m_flat, dp_flat, p_med,
            g1_p, g2_p, gs_p, B, N, num_tiles,
            HAS_MASK=ctx.has_mask, BLOCK_N=ctx.block_med,
        )

        coeff = torch.empty(B, device=device, dtype=torch.float32)
        gmed = torch.empty(B, device=device, dtype=torch.float32)
        ps = torch.empty(B, device=device, dtype=torch.float32)
        _norm_bwd_reduce_kernel[(B,)](
            g1_p, g2_p, gs_p, p_mad, M,
            coeff, gmed, ps, B, num_tiles,
        )

        grad_p = torch.empty((B, N), device=device, dtype=ctx.pred_dtype)
        _norm_bwd_write_kernel[(B, triton.cdiv(N, ctx.block_apply))](
            p_flat, m_flat, dp_flat, grad_p,
            p_med, p_idx, coeff, gmed, ps,
            B, N, HAS_MASK=ctx.has_mask, BLOCK_N=ctx.block_apply,
        )
        return grad_p.view(d_phat.shape), None, None, None, None


class _TritonSSI(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        p_hat: torch.Tensor,
        t_hat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ph = p_hat.contiguous()
        th = t_hat.contiguous()
        N_total = ph.numel()
        has_mask = mask is not None
        m = mask.contiguous() if has_mask else ph

        BLOCK = 2048 if N_total > 1024 else 512
        grid = (triton.cdiv(N_total, BLOCK),)
        out_sum = torch.empty(grid[0], device=ph.device, dtype=torch.float32)
        out_m = torch.empty(grid[0], device=ph.device, dtype=torch.float32) if has_mask else out_sum
        _ssi_fwd_tile_kernel[grid](ph, th, m, out_sum, out_m, N_total, HAS_MASK=has_mask, BLOCK=BLOCK)

        if has_mask:
            denom = out_m.sum() + 1e-8
        else:
            denom = ph.new_empty((), dtype=torch.float32)
        loss = out_sum.sum() / (denom if has_mask else float(max(N_total, 1)))

        ctx.save_for_backward(ph, th, m, denom)
        ctx.has_mask = has_mask
        ctx.N_total = N_total
        ctx.pred_dtype = p_hat.dtype
        return loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        ph, th, m, denom = ctx.saved_tensors
        grad_ph = torch.empty_like(ph, dtype=torch.float32)
        BLOCK = 1024 if ctx.N_total > 512 else 512
        _ssi_bwd_kernel[(triton.cdiv(ctx.N_total, BLOCK),)](
            ph, th, m,
            grad_out.contiguous().view(1), denom.view(1), grad_ph,
            ctx.N_total, float(max(ctx.N_total, 1)),
            HAS_MASK=ctx.has_mask, BLOCK=BLOCK,
        )
        return grad_ph.to(ctx.pred_dtype), None, None


class _TritonTrimmedSSIDense(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        p_hat: torch.Tensor,
        t_hat: torch.Tensor,
        num_keep: int,
    ) -> torch.Tensor:
        B = p_hat.shape[0]
        N = p_hat.numel() // B
        ph_flat = p_hat.contiguous().view(B, N)
        th_flat = t_hat.contiguous().view(B, N)
        device = p_hat.device

        block_med = 2048 if N > 512 else 512
        out_sum = torch.empty(B, device=device, dtype=torch.float32)
        tau = torch.empty(B, device=device, dtype=torch.float32)
        eq_w = torch.empty(B, device=device, dtype=torch.float32)

        _trimmed_ssi_stats_kernel[(B,)](
            ph_flat, th_flat, out_sum, tau, eq_w, B, N, num_keep, BLOCK_N=block_med
        )
        loss = out_sum.sum() / (B * num_keep)

        ctx.save_for_backward(ph_flat, th_flat, tau, eq_w)
        ctx.num_keep = num_keep
        ctx.shape = p_hat.shape
        ctx.dtype = p_hat.dtype
        return loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        ph_flat, th_flat, tau, eq_w = ctx.saved_tensors
        B, N = ph_flat.shape
        grad_ph = torch.empty_like(ph_flat, dtype=torch.float32)
        BLOCK_N = 1024 if N > 512 else 512
        _trimmed_ssi_bwd_kernel[(B, triton.cdiv(N, BLOCK_N))](
            ph_flat, th_flat, tau, eq_w,
            grad_out.contiguous().view(1), grad_ph,
            B, N, 1.0 / (B * ctx.num_keep), BLOCK_N=BLOCK_N,
        )
        return grad_ph.view(ctx.shape).to(ctx.dtype), None, None


class _SortedTritonHDN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        pred: torch.Tensor,
        t_sorted: torch.Tensor,
        t_order: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
        starts: torch.Tensor,
        ends: torch.Tensor,
        t_med_dense: torch.Tensor,
        segments: Tuple[int, ...],
    ) -> torch.Tensor:
        B, N = t_sorted.shape
        G_total = starts.shape[1]
        device = pred.device
        p_flat = pred.contiguous().view(B, N)
        p_sorted = torch.gather(p_flat, 1, t_order)
        has_mask = mask_flat is not None
        m_sorted = torch.gather(mask_flat, 1, t_order) if has_mask else p_sorted

        block_n = 2048 if N > 512 else 512
        p_med = torch.empty((B, G_total), device=device, dtype=torch.float32)
        t_med = torch.empty((B, G_total), device=device, dtype=torch.float32) if has_mask else t_med_dense.contiguous()
        p_mad = torch.empty((B, G_total), device=device, dtype=torch.float32)
        t_mad = torch.empty((B, G_total), device=device, dtype=torch.float32)
        counts = torch.empty((B, G_total), device=device, dtype=torch.float32)
        p_med_idx = torch.empty((B, G_total), device=device, dtype=torch.int32)

        _sorted_seg_stats_kernel[(B, G_total)](
            p_sorted, t_sorted, m_sorted, starts, ends,
            p_med, t_med, p_mad, t_mad, counts, p_med_idx,
            B, N, G_total, HAS_MASK=has_mask, PRE_TMED=False, BLOCK_N=block_n,
        )

        non_flat = ((t_mad > 1e-5) & (counts > 16.0)).to(torch.float32)
        denom = (counts * non_flat).sum() + 1e-8

        out_loss = torch.empty((B, G_total), device=device, dtype=torch.float32)
        sum_s = torch.empty((B, G_total), device=device, dtype=torch.float32)
        sum_sp = torch.empty((B, G_total), device=device, dtype=torch.float32)
        sum_signp = torch.empty((B, G_total), device=device, dtype=torch.float32)

        _sorted_seg_fwd_kernel[(B, G_total)](
            p_sorted, t_sorted, m_sorted, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat,
            out_loss, sum_s, sum_sp, sum_signp,
            B, N, G_total, HAS_MASK=has_mask, BLOCK_N=block_n,
        )
        loss = out_loss.sum() / denom

        ctx.save_for_backward(
            p_sorted, t_sorted, m_sorted, t_order, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat, counts,
            p_med_idx, sum_s, sum_sp, sum_signp, denom,
        )
        ctx.has_mask = has_mask
        ctx.block_n = block_n
        ctx.shape = pred.shape
        ctx.dtype = pred.dtype
        ctx.G_total = G_total
        ctx.segments = segments
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None, None]:
        (
            p_sorted, t_sorted, m_sorted, t_order, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat, counts,
            p_med_idx, sum_s, sum_sp, sum_signp, denom,
        ) = ctx.saved_tensors
        B, N = t_sorted.shape
        mad_clamped = (p_mad < 1e-6).to(torch.float32)
        grad_p_sorted = torch.empty((B, N), device=p_sorted.device, dtype=torch.float32)
        grad_out_1 = grad_out.contiguous().view(1)
        denom_1 = denom.view(1)

        g_offset = 0
        for idx, seg_count in enumerate(ctx.segments):
            _sorted_seg_bwd_kernel[(B, seg_count)](
                p_sorted, t_sorted, m_sorted, grad_p_sorted,
                starts, ends,
                p_med, t_med, p_mad, t_mad,
                mad_clamped, non_flat, counts,
                sum_sp, grad_out_1, denom_1,
                B, N, g_offset, g_offset + seg_count, ctx.G_total,
                HAS_MASK=ctx.has_mask, ACCUMULATE=(idx > 0), BLOCK_N=ctx.block_n,
            )
            g_offset += seg_count

        scale = grad_out.to(torch.float32) / denom
        p_mad_safe = p_mad.clamp(min=1e-6)
        not_clamped = mad_clamped == 0.0
        g_med = scale * non_flat * (
            -sum_s + torch.where(not_clamped, (sum_sp / counts.clamp(min=1.0)) * sum_signp, 0.0)
        ) / p_mad_safe
        valid_med = (p_med_idx >= 0) & (p_med_idx < N) & (non_flat > 0.5)
        idx_safe = torch.where(valid_med, p_med_idx.long(), 0)
        g_med_safe = torch.where(valid_med, g_med, 0.0)
        grad_p_sorted.scatter_add_(1, idx_safe, g_med_safe)

        grad_p = torch.empty_like(grad_p_sorted)
        grad_p.scatter_(1, t_order, grad_p_sorted)
        return grad_p.view(ctx.shape).to(ctx.dtype), None, None, None, None, None, None, None


class _TritonGradNormalScale(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        p: torch.Tensor,
        t: torch.Tensor,
        m: Optional[torch.Tensor],
        normal_beta: float,
        compute_normal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = p.contiguous()
        t = t.contiguous()
        has_mask = m is not None
        m_buf = m.contiguous() if has_mask else p
        B, H, W = p.shape
        BLOCK_W = 256 if W >= 128 else (128 if W >= 64 else (64 if W >= 32 else 32))
        grid = (B * H * triton.cdiv(W, BLOCK_W),)

        out_grad = torch.empty(grid[0], device=p.device, dtype=torch.float32)
        out_norm = torch.empty(grid[0], device=p.device, dtype=torch.float32) if compute_normal else out_grad
        out_dm_grad = torch.empty(grid[0], device=p.device, dtype=torch.float32) if has_mask else out_grad
        out_dm_norm = torch.empty(grid[0], device=p.device, dtype=torch.float32) if (has_mask and compute_normal) else out_grad

        _grad_norm_fwd_kernel[grid](
            p, t, m_buf,
            out_grad, out_norm, out_dm_grad, out_dm_norm,
            B, H, W,
            p.stride(0), p.stride(1), p.stride(2),
            float(normal_beta),
            HAS_MASK=has_mask,
            COMPUTE_NORMAL=compute_normal,
            BLOCK_W=BLOCK_W,
        )

        if has_mask:
            denom_grad = out_dm_grad.sum() + 1e-8
            denom_norm = (out_dm_norm.sum() + 1e-8) if compute_normal else out_grad.new_ones(())
        else:
            denom_grad = out_grad.new_full((), float(B * (H * max(W - 1, 0) + max(H - 1, 0) * W)) + 1e-8)
            denom_norm = out_grad.new_full((), float(B * max(H - 1, 0) * max(W - 1, 0)) + 1e-8) if compute_normal else out_grad.new_ones(())

        grad_loss = out_grad.sum() / denom_grad
        norm_loss = (out_norm.sum() / denom_norm) if compute_normal else p.new_zeros(())

        ctx.save_for_backward(p, t, m_buf, denom_grad, denom_norm)
        ctx.has_mask = has_mask
        ctx.compute_normal = compute_normal
        ctx.normal_beta = float(normal_beta)
        ctx.BLOCK_W = BLOCK_W
        ctx.pred_dtype = p.dtype
        return grad_loss, norm_loss

    @staticmethod
    def backward(
        ctx: Any,
        d_grad: torch.Tensor,
        d_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        p, t, m_buf, denom_grad, denom_norm = ctx.saved_tensors
        B, H, W = p.shape
        w_grad = (d_grad.to(torch.float32) / denom_grad).view(1)
        w_norm = (d_norm.to(torch.float32) / denom_norm).view(1) if ctx.compute_normal else w_grad
        grad_p = torch.empty_like(p, dtype=torch.float32)
        grid = (B * H * triton.cdiv(W, ctx.BLOCK_W),)
        _grad_norm_bwd_kernel[grid](
            p, t, m_buf, grad_p, grad_p,
            w_grad, w_norm,
            B, H, W, 0, 0,
            p.stride(0), p.stride(1), p.stride(2),
            ctx.normal_beta,
            HAS_MASK=ctx.has_mask,
            COMPUTE_NORMAL=ctx.compute_normal,
            ADD_UP=False,
            BLOCK_W=ctx.BLOCK_W,
        )
        return grad_p.to(ctx.pred_dtype), None, None, None, None


class _BucketTritonHDN(torch.autograd.Function):
    """Sort-free 4-bucket contiguous HDN for dense (2, 4) segments."""

    @staticmethod
    def forward(
        ctx: Any,
        pred: torch.Tensor,
        target: torch.Tensor,
        q3: torch.Tensor,
        strict_lt: bool,
    ) -> torch.Tensor:
        B, H, W = pred.shape
        N = H * W
        device = pred.device
        p_flat = pred.contiguous().view(B, N)
        t_flat = target.contiguous().view(B, N)

        block_n = 2048 if N > 512 else 512
        num_tiles = triton.cdiv(N, block_n)
        tile_cnt = torch.empty((B, num_tiles, 4), device=device, dtype=torch.int32)
        starts = torch.empty((B, 6), device=device, dtype=torch.int32)
        ends = torch.empty((B, 6), device=device, dtype=torch.int32)
        _bucket4_count_kernel[(B,)](
            t_flat, q3, tile_cnt, starts, ends,
            B, N, num_tiles, STRICT_LT=strict_lt, BLOCK_N=block_n,
        )

        p_part = torch.empty((B, N), device=device, dtype=pred.dtype)
        t_part = torch.empty((B, N), device=device, dtype=torch.float32)
        orig_idx = torch.empty((B, N), device=device, dtype=torch.int32)
        _bucket4_scatter_kernel[(B, num_tiles)](
            p_flat, t_flat, q3, tile_cnt, starts,
            p_part, t_part, orig_idx,
            B, N, num_tiles, STRICT_LT=strict_lt, BLOCK_N=block_n,
        )

        t_med = torch.empty((B, 6), device=device, dtype=torch.float32)
        _seg_t_med_kernel[(B, 6)](t_part, starts, ends, t_med, B, N, 6, BLOCK_N=block_n)

        p_med = torch.empty((B, 6), device=device, dtype=torch.float32)
        p_mad = torch.empty((B, 6), device=device, dtype=torch.float32)
        t_mad = torch.empty((B, 6), device=device, dtype=torch.float32)
        counts = torch.empty((B, 6), device=device, dtype=torch.float32)
        p_med_idx = torch.empty((B, 6), device=device, dtype=torch.int32)
        _sorted_seg_stats_kernel[(B, 6)](
            p_part, t_part, p_part, starts, ends,
            p_med, t_med, p_mad, t_mad, counts, p_med_idx,
            B, N, 6, HAS_MASK=False, PRE_TMED=True, BLOCK_N=block_n,
        )

        non_flat = ((t_mad > 1e-5) & (counts > 16.0)).to(torch.float32)
        denom = (counts * non_flat).sum() + 1e-8

        out_loss = torch.empty((B, 6), device=device, dtype=torch.float32)
        sum_s = torch.empty((B, 6), device=device, dtype=torch.float32)
        sum_sp = torch.empty((B, 6), device=device, dtype=torch.float32)
        sum_signp = torch.empty((B, 6), device=device, dtype=torch.float32)
        _sorted_seg_fwd_kernel[(B, 6)](
            p_part, t_part, p_part, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat,
            out_loss, sum_s, sum_sp, sum_signp,
            B, N, 6, HAS_MASK=False, BLOCK_N=block_n,
        )
        loss = out_loss.sum() / denom

        ctx.save_for_backward(
            p_part, t_part, orig_idx, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat, counts,
            p_med_idx, sum_s, sum_sp, sum_signp, denom,
        )
        ctx.block_n = block_n
        ctx.shape = pred.shape
        ctx.dtype = pred.dtype
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None]:
        (
            p_part, t_part, orig_idx, starts, ends,
            p_med, t_med, p_mad, t_mad, non_flat, counts,
            p_med_idx, sum_s, sum_sp, sum_signp, denom,
        ) = ctx.saved_tensors
        B, N = t_part.shape
        grad_p = torch.empty((B, N), device=p_part.device, dtype=ctx.dtype)
        _bucket_bwd_scatter_kernel[(B, 4)](
            p_part, t_part, orig_idx, grad_p,
            starts, ends,
            p_med, t_med, p_mad, t_mad,
            non_flat, counts, p_med_idx,
            sum_s, sum_sp, sum_signp,
            grad_out.contiguous().view(1), denom.view(1),
            B, N, BLOCK_N=ctx.block_n,
        )
        return grad_p.view(ctx.shape), None, None, None


class _TritonMultiScaleGradNormal(torch.autograd.Function):
    """Fused multi-scale gradient + normal loss skipping avg_pool2d_backward."""

    @staticmethod
    def forward(
        ctx: Any,
        p_hat: torch.Tensor,
        t_hat: torch.Tensor,
        mask: Optional[torch.Tensor],
        grad_scales: int,
        normal_beta: float,
        compute_normal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p_hat.shape[0]
        device = p_hat.device
        has_mask = mask is not None

        p_curr = p_hat.contiguous().float()
        t_curr = t_hat.contiguous().float()
        m_curr = mask.contiguous().float() if has_mask else p_curr

        ps = [p_curr]
        ts = [t_curr]
        ms = [m_curr]
        for _ in range(1, grad_scales):
            _, h_i, w_i = ps[-1].shape
            if h_i < 2 or w_i < 2:
                break
            h_o, w_o = h_i // 2, w_i // 2
            p_next = torch.empty((B, h_o, w_o), device=device, dtype=torch.float32)
            t_next = torch.empty((B, h_o, w_o), device=device, dtype=torch.float32)
            m_next = torch.empty((B, h_o, w_o), device=device, dtype=torch.float32) if has_mask else p_next
            bx = 128 if w_o >= 128 else (64 if w_o >= 64 else 32)
            _pool2x_pair_kernel[(B * h_o * triton.cdiv(w_o, bx),)](
                ps[-1], ts[-1], ms[-1], p_next, t_next, m_next,
                B, h_i, w_i, h_o, w_o, HAS_MASK=has_mask, BLOCK_X=bx,
            )
            ps.append(p_next)
            ts.append(t_next)
            ms.append(m_next)

        actual_scales = len(ps)
        bws = []
        denom_grads = []
        denom_norms = []
        grad_terms = []
        norm_terms = []

        for s in range(actual_scales):
            pi, ti, mi = ps[s], ts[s], ms[s]
            _, hs, ws = pi.shape
            bw = 256 if ws >= 128 else (128 if ws >= 64 else (64 if ws >= 32 else 32))
            bws.append(bw)
            grid = (B * hs * triton.cdiv(ws, bw),)
            out_g = torch.empty(grid[0], device=device, dtype=torch.float32)
            out_n = torch.empty(grid[0], device=device, dtype=torch.float32) if compute_normal else out_g
            out_dmg = torch.empty(grid[0], device=device, dtype=torch.float32) if has_mask else out_g
            out_dmn = torch.empty(grid[0], device=device, dtype=torch.float32) if (has_mask and compute_normal) else out_g

            _grad_norm_fwd_kernel[grid](
                pi, ti, mi, out_g, out_n, out_dmg, out_dmn,
                B, hs, ws, pi.stride(0), pi.stride(1), pi.stride(2),
                float(normal_beta), HAS_MASK=has_mask, COMPUTE_NORMAL=compute_normal, BLOCK_W=bw,
            )
            if has_mask:
                dg = out_dmg.sum() + 1e-8
                dn = (out_dmn.sum() + 1e-8) if compute_normal else out_g.new_ones(())
            else:
                dg = out_g.new_full((), float(B * (hs * max(ws - 1, 0) + max(hs - 1, 0) * ws)) + 1e-8)
                dn = out_g.new_full((), float(B * max(hs - 1, 0) * max(ws - 1, 0)) + 1e-8) if compute_normal else out_g.new_ones(())
            denom_grads.append(dg)
            denom_norms.append(dn)
            grad_terms.append(out_g.sum() / dg)
            if compute_normal:
                norm_terms.append(out_n.sum() / dn)

        grad_total = torch.stack(grad_terms).sum() / actual_scales
        norm_total = (torch.stack(norm_terms).sum() / actual_scales) if compute_normal else p_hat.new_zeros((), dtype=torch.float32)

        ctx.save_for_backward(*ps, *ts, *ms, *denom_grads, *denom_norms)
        ctx.actual_scales = actual_scales
        ctx.bws = bws
        ctx.has_mask = has_mask
        ctx.compute_normal = compute_normal
        ctx.normal_beta = float(normal_beta)
        ctx.pred_dtype = p_hat.dtype
        return grad_total, norm_total

    @staticmethod
    def backward(
        ctx: Any,
        d_grad: torch.Tensor,
        d_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None, None]:
        S = ctx.actual_scales
        saved = ctx.saved_tensors
        ps = saved[0:S]
        ts = saved[S:2 * S]
        ms = saved[2 * S:3 * S]
        denom_grads = saved[3 * S:4 * S]
        denom_norms = saved[4 * S:5 * S]

        B = ps[0].shape[0]
        dg_f = d_grad.to(torch.float32) / float(S)
        dn_f = d_norm.to(torch.float32) / float(S)

        grad_up: Optional[torch.Tensor] = None
        for s in range(S - 1, -1, -1):
            pi, ti, mi = ps[s], ts[s], ms[s]
            _, hs, ws = pi.shape
            bw = ctx.bws[s]
            wg = (dg_f / denom_grads[s]).view(1)
            wn = (dn_f / denom_norms[s]).view(1) if ctx.compute_normal else wg
            gp = torch.empty((B, hs, ws), device=pi.device, dtype=torch.float32)
            h_up, w_up = (grad_up.shape[1], grad_up.shape[2]) if grad_up is not None else (0, 0)
            _grad_norm_bwd_kernel[(B * hs * triton.cdiv(ws, bw),)](
                pi, ti, mi, grad_up if grad_up is not None else gp, gp,
                wg, wn,
                B, hs, ws, h_up, w_up,
                pi.stride(0), pi.stride(1), pi.stride(2),
                ctx.normal_beta,
                HAS_MASK=ctx.has_mask,
                COMPUTE_NORMAL=ctx.compute_normal,
                ADD_UP=(grad_up is not None),
                BLOCK_W=bw,
            )
            grad_up = gp

        assert grad_up is not None
        return grad_up.to(ctx.pred_dtype), None, None, None, None, None


# ---------------------------------------------------------------------------
# Public Entry Points (Triton on CUDA with fast pure-PyTorch fallbacks)
# ---------------------------------------------------------------------------

def normalize_depth(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    t_med_pre: Optional[torch.Tensor] = None,
    mad_denom_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Median/MAD scale-and-shift normalization for [B, H, W] predictions and targets."""
    if mask is not None and not mask.is_floating_point():
        mask = mask.float()
    if can_use_triton(pred):
        return _TritonNormalize.apply(pred, target, mask, t_med_pre, float(mad_denom_eps))

    p = pred.float()
    t = target.float()
    B, H, W = p.shape
    N = H * W

    if mask is None:
        k = (N - 1) // 2 + 1
        p_med = torch.kthvalue(p.reshape(B, N), k, dim=1).values.view(B, 1, 1)
        if t_med_pre is None:
            t_med = torch.kthvalue(t.reshape(B, N), k, dim=1).values.view(B, 1, 1)
        else:
            t_med = t_med_pre.float().view(B, 1, 1)
        denom = max(float(N) + float(mad_denom_eps), 1e-8)
        p_mad = (p - p_med).abs().sum(dim=(1, 2), keepdim=True).div_(denom).clamp_(min=1e-6)
        t_mad = (t - t_med).abs().sum(dim=(1, 2), keepdim=True).div_(denom).clamp_(min=1e-6)
        return (p - p_med) / p_mad, (t - t_med) / t_mad

    m = mask.float()
    valid = m > 0.5
    p_nan = p.masked_fill(~valid, float("nan")).reshape(B, N)
    t_nan = t.masked_fill(~valid, float("nan")).reshape(B, N)
    p_med = torch.nanmedian(p_nan, dim=1).values.nan_to_num(nan=0.0).view(B, 1, 1)
    t_med = torch.nanmedian(t_nan, dim=1).values.nan_to_num(nan=0.0).view(B, 1, 1)
    denom = (m.sum(dim=(1, 2), keepdim=True) + float(mad_denom_eps)).clamp(min=1e-8)
    p_mad = ((p - p_med) * m).abs().sum(dim=(1, 2), keepdim=True).div(denom).clamp(min=1e-6)
    t_mad = ((t - t_med) * m).abs().sum(dim=(1, 2), keepdim=True).div(denom).clamp(min=1e-6)
    return (p - p_med) / p_mad, (t - t_med) / t_mad


def ssi_loss(
    p_hat: torch.Tensor,
    t_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean L1 loss on normalized predictions and targets (dense or masked)."""
    if mask is not None and not mask.is_floating_point():
        mask = mask.float()
    if can_use_triton(p_hat):
        return _TritonSSI.apply(p_hat, t_hat, mask)
    diff = (p_hat.float() - t_hat.float()).abs()
    if mask is None:
        return diff.mean()
    m = mask.float()
    return (diff * m).sum() / (m.sum() + 1e-8)


def trimmed_ssi_loss(
    p_hat: torch.Tensor,
    t_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    trim_ratio: float = 0.10,
) -> torch.Tensor:
    """Trimmed L1 SSI loss dropping the top-trim_ratio largest pixel errors per image."""
    B = p_hat.shape[0]
    N = p_hat.numel() // B

    if mask is None:
        num_keep = max(int(N * (1.0 - trim_ratio)), 1) if trim_ratio > 0.0 else N
        if num_keep >= N:
            return ssi_loss(p_hat, t_hat, None)
        if can_use_triton(p_hat):
            return _TritonTrimmedSSIDense.apply(p_hat, t_hat, num_keep)
        diff = (p_hat.float() - t_hat.float()).abs().reshape(B, N)
        kept, _ = torch.topk(diff, num_keep, dim=-1, largest=False)
        return kept.mean()

    if not mask.is_floating_point():
        mask = mask.float()

    # Masked trimmed SSI: trim top-trim_ratio errors among valid pixels per image
    m_flat = mask.float().reshape(B, N)
    diff = (p_hat.float() - t_hat.float()).abs().reshape(B, N)
    if trim_ratio <= 0.0:
        return (diff * m_flat).sum() / (m_flat.sum() + 1e-8)
    valid_counts = (m_flat >= 0.5).sum(dim=1)
    keep_counts = (valid_counts.float() * (1.0 - trim_ratio)).long().clamp(min=1)
    diff_masked = diff.masked_fill(m_flat < 0.5, float("inf"))
    sorted_diff, _ = torch.sort(diff_masked, dim=1)
    idx = torch.arange(N, device=p_hat.device).unsqueeze(0)
    keep_mask = (idx < keep_counts.unsqueeze(1)).float()
    safe_diff = torch.where(torch.isfinite(sorted_diff), sorted_diff, torch.zeros_like(sorted_diff))
    return (safe_diff * keep_mask).sum() / (keep_counts.sum().float() + 1e-8)


def compute_hdn_sort_plan(
    target_flat: torch.Tensor,
    segments: Sequence[int],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort target once per batch to produce shared t_med and segment slice bounds."""
    B, N = target_flat.shape
    device = target_flat.device
    t_sorted, t_order = torch.sort(target_flat.float(), dim=1)
    t_med_global = t_sorted[:, (N - 1) // 2]

    bounds = []
    if mode == "dp":
        for S in segments:
            for s in range(1, S):
                pos = (s / S) * (N - 1)
                lo = int(math.floor(pos))
                hi = int(math.ceil(pos))
                bounds.append(t_sorted[:, lo] if lo == hi else torch.lerp(t_sorted[:, lo], t_sorted[:, hi], pos - lo))
    else:
        t_min = t_sorted[:, 0]
        t_max = t_sorted[:, -1]
        span = (t_max - t_min).clamp(min=1e-6)
        for S in segments:
            for s in range(1, S):
                bounds.append(t_min + (s / S) * span)

    starts = []
    ends = []
    zeros = torch.zeros(B, device=device, dtype=torch.int32)
    full_n = torch.full((B,), N, device=device, dtype=torch.int32)

    if bounds:
        b_idx = torch.searchsorted(
            t_sorted,
            torch.stack(bounds, dim=1).contiguous(),
            right=(mode == "dp"),
        ).to(torch.int32)
        col = 0
        for S in segments:
            prev = zeros
            for s in range(S):
                nxt = b_idx[:, col] if s < S - 1 else full_n
                if s < S - 1:
                    col += 1
                starts.append(prev)
                ends.append(nxt)
                prev = nxt
    else:
        for _ in segments:
            starts.append(zeros)
            ends.append(full_n)

    st = torch.stack(starts, dim=1).contiguous()
    en = torch.stack(ends, dim=1).contiguous()
    cnt = (en - st).clamp(min=1)
    t_med_seg = torch.where(
        en > st,
        torch.gather(t_sorted, 1, (st + (cnt - 1) // 2).long().clamp(max=N - 1)),
        0.0,
    )
    return t_sorted, t_order, t_med_global, st, en, t_med_seg


def compute_hdn_thresholds(
    target_flat: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute global target median and 4-bin HDN thresholds (dp/dr) without sorting."""
    B, N = target_flat.shape
    t_flat = target_flat.contiguous().float()
    q3 = torch.empty((B, 3), device=target_flat.device, dtype=torch.float32)
    block_n = 2048 if N > 512 else 512
    if mode == "dp":
        k0 = (N - 1) // 4
        k1 = (N - 1) // 2
        k2 = (3 * (N - 1)) // 4
        _q3_packed1024_kernel[(B,)](t_flat, q3, B, N, k0, k1, k2, BLOCK_N=block_n)
        t_med = q3[:, 1].contiguous()
    else:
        _dr_bounds_kernel[(B,)](t_flat, q3, B, N, BLOCK_N=block_n)
        t_med = torch.empty(B, device=target_flat.device, dtype=torch.float32)
        p_dummy = torch.empty(B, device=target_flat.device, dtype=torch.int64)
        _dense_median_kernel[(B,)](t_flat, t_med, p_dummy, B, N, FIND_IDX=False, BLOCK_N=block_n)
    return t_med, q3


def hdn_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    segments: Sequence[int] = (2, 4),
    mode: str = "dp",
    sort_plan: Optional[Tuple[torch.Tensor, ...]] = None,
    bucket_q3: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-level Hybrid Depth Normalization loss with optional precomputed plan/thresholds."""
    if not segments:
        return pred.new_zeros((), dtype=torch.float32)
    if mask is not None and not mask.is_floating_point():
        mask = mask.float()

    B, H, W = pred.shape
    N = H * W

    if can_use_triton(pred):
        if mask is None and tuple(segments) == (2, 4) and sort_plan is None and N >= 64:
            if bucket_q3 is None:
                _, bucket_q3 = compute_hdn_thresholds(target.contiguous().view(B, N), mode)
            return _BucketTritonHDN.apply(pred, target, bucket_q3, (mode == "dr"))
        if sort_plan is None:
            sort_plan = compute_hdn_sort_plan(target.contiguous().view(B, N), segments, mode)
        t_sorted, t_order, _, starts, ends, t_med_dense = sort_plan
        mask_flat = mask.contiguous().view(B, N) if mask is not None else None
        return _SortedTritonHDN.apply(pred, t_sorted, t_order, mask_flat, starts, ends, t_med_dense, tuple(segments))

    # Fast pure-PyTorch fallback (CPU / non-Triton): avoids redundant nanmedian/clones
    p_flat = pred.float().reshape(B, N)
    t_flat = target.float().reshape(B, N)
    m_flat = mask.float().reshape(B, N) if mask is not None else None
    t_sorted, t_order, _, starts, ends, t_med_dense = (
        sort_plan if sort_plan is not None else compute_hdn_sort_plan(t_flat, segments, mode)
    )
    p_sorted = torch.gather(p_flat, 1, t_order)
    m_sorted = torch.gather(m_flat, 1, t_order) if m_flat is not None else None

    total_loss = pred.new_zeros((), dtype=torch.float32)
    total_denom = pred.new_zeros((), dtype=torch.float32)
    G_total = starts.shape[1]

    for b in range(B):
        pb = p_sorted[b]
        tb = t_sorted[b]
        mb = m_sorted[b] if m_sorted is not None else None
        for g in range(G_total):
            st_i = int(starts[b, g].item())
            en_i = int(ends[b, g].item())
            if en_i <= st_i:
                continue
            ps = pb[st_i:en_i]
            ts = tb[st_i:en_i]
            if mb is not None:
                ms = mb[st_i:en_i]
                ok = ms >= 0.5
                cnt = int(ok.sum().item())
                if cnt <= 16:
                    continue
                ps_v = ps[ok]
                ts_v = ts[ok]
            else:
                cnt = en_i - st_i
                if cnt <= 16:
                    continue
                ps_v = ps
                ts_v = ts
            k = (cnt - 1) // 2 + 1
            t_med = ts_v[k - 1]
            t_mad = (ts_v - t_med).abs().mean()
            if float(t_mad.item()) <= 1e-5:
                continue
            p_med = torch.kthvalue(ps_v, k).values
            p_mad = (ps_v - p_med).abs().mean().clamp(min=1e-6)
            total_loss = total_loss + ((ps_v - p_med) / p_mad - (ts_v - t_med) / t_mad.clamp(min=1e-6)).abs().sum()
            total_denom = total_denom + cnt

    return total_loss / (total_denom + 1e-8)


def grad_normal_loss(
    p_hat: torch.Tensor,
    t_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    grad_scales: int = 4,
    normal_beta: float = 1.0,
    compute_normal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-scale gradient L1 and optional intrinsics-free virtual surface normal cosine loss."""
    if can_use_triton(p_hat) and grad_scales >= 1 and (mask is None or grad_scales <= 4):
        m_in = (mask if mask.is_floating_point() else mask.float()) if mask is not None else None
        return _TritonMultiScaleGradNormal.apply(
            p_hat, t_hat, m_in, int(grad_scales), float(normal_beta), bool(compute_normal)
        )

    total_grad = p_hat.new_zeros((), dtype=torch.float32)
    total_norm = p_hat.new_zeros((), dtype=torch.float32)
    use_triton = can_use_triton(p_hat)

    p = p_hat if p_hat.dtype == torch.float32 else p_hat.float()
    t = t_hat if t_hat.dtype == torch.float32 else t_hat.float()
    m = (mask if mask.dtype == torch.float32 else mask.float()) if mask is not None else None
    actual_scales = 0

    for scale in range(grad_scales):
        if scale > 0:
            if p.shape[-2] < 2 or p.shape[-1] < 2:
                break
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            t = F.avg_pool2d(t.unsqueeze(1), 2).squeeze(1)
            if mask is not None:
                if scale < 4:
                    m = (F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1) > 0.99).to(torch.float32)
                else:
                    m = (F.avg_pool2d(mask.unsqueeze(1).float(), 2 ** scale).squeeze(1) > 0.99).to(torch.float32)

        actual_scales += 1
        if use_triton:
            g_l, n_l = _TritonGradNormalScale.apply(p, t, m, normal_beta, compute_normal)
        else:
            B_s, H_s, W_s = p.shape
            p_dx = p[:, :, 1:] - p[:, :, :-1]
            t_dx = t[:, :, 1:] - t[:, :, :-1]
            p_dy = p[:, 1:, :] - p[:, :-1, :]
            t_dy = t[:, 1:, :] - t[:, :-1, :]

            if m is not None:
                m_x = m[:, :, 1:] * m[:, :, :-1]
                m_y = m[:, 1:, :] * m[:, :-1, :]
                denom_g = m_x.sum() + m_y.sum() + 1e-8
                g_l = ((p_dx - t_dx).abs() * m_x).sum() + ((p_dy - t_dy).abs() * m_y).sum()
                g_l = g_l / denom_g
            else:
                denom_g = float(B_s * (H_s * max(W_s - 1, 0) + max(H_s - 1, 0) * W_s)) + 1e-8
                g_l = ((p_dx - t_dx).abs().sum() + (p_dy - t_dy).abs().sum()) / denom_g

            if compute_normal:
                px = normal_beta * p_dx[:, :-1, :]
                py = normal_beta * p_dy[:, :, :-1]
                tx = normal_beta * t_dx[:, :-1, :]
                ty = normal_beta * t_dy[:, :, :-1]
                rp = torch.rsqrt(px * px + py * py + 1.0)
                rt = torch.rsqrt(tx * tx + ty * ty + 1.0)
                cos_sim = (px * tx + py * ty + 1.0) * rp * rt
                if m is not None:
                    m_xy = m_x[:, :-1, :] * m_y[:, :, :-1]
                    n_l = ((1.0 - cos_sim) * m_xy).sum() / (m_xy.sum() + 1e-8)
                else:
                    denom_n = float(B_s * max(H_s - 1, 0) * max(W_s - 1, 0)) + 1e-8
                    n_l = (1.0 - cos_sim).sum() / denom_n
            else:
                n_l = p.new_zeros(())

        total_grad = total_grad + g_l
        if compute_normal:
            total_norm = total_norm + n_l

    denom_scales = max(actual_scales, 1)
    if compute_normal:
        return total_grad / denom_scales, total_norm / denom_scales
    return total_grad / denom_scales, p_hat.new_zeros((), dtype=torch.float32)
