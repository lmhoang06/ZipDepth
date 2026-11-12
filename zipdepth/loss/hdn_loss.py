"""
Shared-Context Hybrid Depth Normalization Distillation Loss for ZipDepth.

Integrates:
  1. Trimmed Global SSI Loss (MiDaS / DepthAnythingV2 top-10% outlier trimming)
  2. Shared-Context Hybrid Depth Normalization Loss L_sc (Distill Any Depth, arXiv:2502.19204)
  3. Multi-Scale Gradient + 2D Disparity Surface Normal Cosine Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDNDistillationLoss(nn.Module):
    """
    Upgraded distillation loss for offline dense pseudo-labels.

    Combines:
      - Global Trimmed SSI (S=1 context, rejects top-10% teacher hallucinations)
      - Hybrid Depth Normalization across depth segments (S in {2, 4}, HDN-DP)
      - Multi-scale gradient matching across 4 pooled resolutions
      - 2D disparity virtual surface normal cosine alignment
    """

    def __init__(
        self,
        alpha_ssi=1.0,
        alpha_hdn=1.0,
        alpha_grad=2.0,
        alpha_normal=0.5,
        grad_scales=4,
        trim_ratio=0.10,
        hdn_segments=(2, 4),
        hdn_mode="dp",
        normal_beta=1.0,
    ):
        super().__init__()
        self.alpha = {
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

    def forward(self, pred, target, mask=None):
        pred, target = self._squeeze(pred, target)
        dense_mode = mask is None
        if dense_mode:
            mask = torch.ones_like(target, dtype=torch.float32)
        else:
            (mask,) = self._squeeze(mask)
            mask = mask.float()

        # 1. Global Normalization (S = 1 context)
        p_hat, t_hat = self._normalize_global(pred, target, mask, dense_mode=dense_mode)

        # 2. Trimmed Global SSI Loss (rejects top 10% noisy pseudo-label outliers per image)
        ssi_loss = self._trimmed_ssi_loss(p_hat, t_hat, mask, dense_mode=dense_mode)

        # 3. Distill Any Depth Hybrid Normalization Loss (L_sc across S in {2, 4} depth segments)
        if self.alpha["hdn"] > 0 and len(self.hdn_segments) > 0:
            hdn_loss = self._hybrid_depth_norm_loss(pred, target, mask)
        else:
            hdn_loss = torch.zeros_like(ssi_loss)

        # 4. Multi-Scale Gradient + 2D Disparity Normal Loss (reuses p_dx, p_dy, t_dx, t_dy)
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

    # ------------------------------------------------------------------
    def _normalize_global(self, pred, target, mask, dense_mode=True):
        B = pred.shape[0]
        if dense_mode:
            # Fast path for dense pseudo-labels: avoids .clone() and slow torch.nanmedian
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

        p_hat = (pred - p_med) / p_mad.clamp(min=1e-6)
        t_hat = (target - t_med) / t_mad.clamp(min=1e-6)
        return p_hat, t_hat

    def _trimmed_ssi_loss(self, p_hat, t_hat, mask, dense_mode=True):
        B = p_hat.shape[0]
        diff = (p_hat - t_hat).abs()

        if self.trim_ratio <= 0.0:
            M = mask.sum() + 1e-8
            return (diff * mask).sum() / M

        if dense_mode:
            flat_diff = diff.view(B, -1)
            num_keep = int(flat_diff.shape[1] * (1.0 - self.trim_ratio))
            # Keep the smallest (1 - trim_ratio) residuals per image via topk(largest=False)
            kept_diff, _ = torch.topk(flat_diff, k=num_keep, dim=1, largest=False, sorted=False)
            return kept_diff.mean()
        else:
            # Masked trimming fallback
            diff_masked = diff.masked_fill(mask < 0.5, float("inf")).view(B, -1)
            sorted_diff, _ = torch.sort(diff_masked, dim=1)
            valid_counts = (mask.view(B, -1) >= 0.5).sum(dim=1)
            keep_counts = (valid_counts.float() * (1.0 - self.trim_ratio)).long().clamp(min=1)
            idx = torch.arange(diff_masked.shape[1], device=p_hat.device).unsqueeze(0)
            keep_mask = (idx < keep_counts.unsqueeze(1)).float()
            safe_diff = torch.where(torch.isfinite(sorted_diff), sorted_diff, torch.zeros_like(sorted_diff))
            return (safe_diff * keep_mask).sum() / (keep_counts.sum().float() + 1e-8)

    def _hybrid_depth_norm_loss(self, pred, target, mask):
        """
        Distill Any Depth (arXiv:2502.19204) / HDN (NeurIPS 2022) Shared-Context Normalization.
        Partitions the teacher's depth map `target` into S in {2, 4} depth segments and normalizes
        pred and target independently within each depth segment.
        """
        B, H, W = target.shape
        t_flat = target.view(B, -1)
        seg_masks = []

        if self.hdn_mode == "dp":
            # Depth Percentile (Quantile) mode: guarantees balanced pixel counts per segment
            for S in self.hdn_segments:
                q_probs = torch.linspace(0.0, 1.0, steps=S + 1, device=target.device, dtype=target.dtype)
                # quantiles shape: [S + 1, B, 1, 1]
                q_vals = torch.quantile(t_flat, q_probs, dim=1).view(S + 1, B, 1, 1)
                for s in range(S):
                    low, high = q_vals[s], q_vals[s + 1]
                    if s == 0:
                        m_u = (target >= low) & (target <= high)
                    else:
                        m_u = (target > low) & (target <= high)
                    seg_masks.append(m_u.float() * mask)
        else:
            # Depth Range (HDN-DR) mode: uniform intervals between [t_min, t_max]
            t_min = t_flat.min(dim=1).values.view(B, 1, 1)
            t_max = t_flat.max(dim=1).values.view(B, 1, 1)
            t_span = (t_max - t_min).clamp(min=1e-6)
            for S in self.hdn_segments:
                step = t_span / S
                for s in range(S):
                    low = t_min + s * step
                    high = t_min + (s + 1) * step
                    if s == S - 1:
                        m_u = (target >= low) & (target <= high + 1e-5)
                    else:
                        m_u = (target >= low) & (target < high)
                    seg_masks.append(m_u.float() * mask)

        # Stack all G segment masks: [B, G, H*W]
        M_u = torch.stack(seg_masks, dim=1).view(B, -1, H * W)
        G = M_u.shape[1]
        p_exp = pred.view(B, 1, H * W).expand(B, G, H * W)
        t_exp = target.view(B, 1, H * W).expand(B, G, H * W)

        # Compute masked median per (Batch, Segment)
        p_nan = p_exp.masked_fill(M_u < 0.5, float("nan"))
        t_nan = t_exp.masked_fill(M_u < 0.5, float("nan"))
        p_med = torch.nanmedian(p_nan, dim=2, keepdim=True).values.nan_to_num(0.0)
        t_med = torch.nanmedian(t_nan, dim=2, keepdim=True).values.nan_to_num(0.0)

        counts = M_u.sum(dim=2, keepdim=True)
        valid_group = (counts > 16.0).float()  # Ignore degenerate empty segments
        safe_counts = counts.clamp(min=1.0)

        p_mad = ((p_exp - p_med).abs() * M_u).sum(dim=2, keepdim=True) / safe_counts
        t_mad = ((t_exp - t_med).abs() * M_u).sum(dim=2, keepdim=True) / safe_counts

        # Skip segments where teacher depth is completely flat (e.g., pure 0.0 sky bin)
        non_flat = (t_mad > 1e-5).float() * valid_group

        p_norm = (p_exp - p_med) / p_mad.clamp(min=1e-6)
        t_norm = (t_exp - t_med) / t_mad.clamp(min=1e-6)

        diff = (p_norm - t_norm).abs() * M_u * non_flat
        denom = (M_u * non_flat).sum() + 1e-8
        return diff.sum() / denom

    def _gradient_and_normal_loss(self, pred, target, mask):
        total_grad = 0.0
        total_normal = 0.0

        for scale in range(self.grad_scales):
            k = 2 ** scale
            if k > 1:
                p = F.avg_pool2d(pred.unsqueeze(1), k).squeeze(1)
                t = F.avg_pool2d(target.unsqueeze(1), k).squeeze(1)
                m = (F.avg_pool2d(mask.unsqueeze(1), k).squeeze(1) > 0.99).float()
            else:
                p, t, m = pred, target, mask

            p_dx = p[:, :, 1:] - p[:, :, :-1]
            p_dy = p[:, 1:, :] - p[:, :-1, :]
            t_dx = t[:, :, 1:] - t[:, :, :-1]
            t_dy = t[:, 1:, :] - t[:, :-1, :]

            m_x = m[:, :, 1:] * m[:, :, :-1]
            m_y = m[:, 1:, :] * m[:, :-1, :]

            loss_x = ((p_dx - t_dx).abs() * m_x).sum()
            loss_y = ((p_dy - t_dy).abs() * m_y).sum()
            denom = m_x.sum() + m_y.sum() + 1e-8
            total_grad += (loss_x + loss_y) / denom

            # 2D Disparity Virtual Normal Cosine Loss on overlapping valid (H-1, W-1) grid
            if self.alpha["normal"] > 0:
                px = self.normal_beta * p_dx[:, :-1, :]
                py = self.normal_beta * p_dy[:, :, :-1]
                tx = self.normal_beta * t_dx[:, :-1, :]
                ty = self.normal_beta * t_dy[:, :, :-1]
                m_xy = m_x[:, :-1, :] * m_y[:, :, :-1]

                # Surface normal n = (-beta*dx, -beta*dy, 1) / ||(-beta*dx, -beta*dy, 1)||
                ones = torch.ones_like(px)
                n_pred = F.normalize(torch.stack([-px, -py, ones], dim=1), dim=1, eps=1e-6)
                n_tgt = F.normalize(torch.stack([-tx, -ty, ones], dim=1), dim=1, eps=1e-6)

                cos_sim = (n_pred * n_tgt).sum(dim=1)
                normal_err = (1.0 - cos_sim) * m_xy
                total_normal += normal_err.sum() / (m_xy.sum() + 1e-8)

        return total_grad / self.grad_scales, total_normal / self.grad_scales

    @staticmethod
    def _squeeze(*tensors):
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)

