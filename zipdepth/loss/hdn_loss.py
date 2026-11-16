"""
Shared-Context Hybrid Depth Normalization Distillation Loss for ZipDepth.

Integrates:
  1. Trimmed Global SSI Loss (MiDaS / DepthAnythingV2 top-10% outlier trimming)
  2. Shared-Context Hybrid Depth Normalization Loss L_sc (Distill Any Depth, arXiv:2502.19204)
  3. Multi-Scale Gradient + 2D Disparity Surface Normal Cosine Loss
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipdepth.loss._triton_ops import (
    compute_hdn_sort_plan,
    grad_normal_loss,
    hdn_loss as fast_hdn_loss,
    normalize_depth,
    trimmed_ssi_loss,
)


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
        self.hdn_segments: Tuple[int, ...] = tuple(int(s) for s in hdn_segments)
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
        if not dense_mode:
            (mask,) = self._squeeze(mask)
            mask = mask if mask.is_floating_point() else mask.float()

        use_hdn = self.alpha["hdn"] > 0 and len(self.hdn_segments) > 0
        sort_plan: Optional[Tuple[torch.Tensor, ...]] = None
        t_med_pre: Optional[torch.Tensor] = None

        # Pre-sort target once per batch to share t_med and HDN slice boundaries
        if use_hdn:
            B, H, W = target.shape
            sort_plan = compute_hdn_sort_plan(
                target.contiguous().view(B, H * W),
                self.hdn_segments,
                self.hdn_mode,
            )
            if dense_mode:
                t_med_pre = sort_plan[2]

        # 1. Global Normalization (S = 1 context)
        p_hat, t_hat = normalize_depth(
            pred,
            target,
            mask=None if dense_mode else mask,
            t_med_pre=t_med_pre,
            mad_denom_eps=0.0,
        )

        # 2. Trimmed Global SSI Loss (rejects top trim_ratio noisy pseudo-label outliers per image)
        ssi_loss = trimmed_ssi_loss(
            p_hat,
            t_hat,
            mask=None if dense_mode else mask,
            trim_ratio=self.trim_ratio,
        )

        # 3. Distill Any Depth Hybrid Normalization Loss (L_sc across S depth segments)
        if use_hdn:
            hdn_loss = fast_hdn_loss(
                pred,
                target,
                mask=None if dense_mode else mask,
                segments=self.hdn_segments,
                mode=self.hdn_mode,
                sort_plan=sort_plan,
            )
        else:
            hdn_loss = pred.new_zeros((), dtype=torch.float32)

        # 4. Multi-Scale Gradient + 2D Disparity Virtual Normal Loss
        grad_loss, normal_loss = self._gradient_and_normal_loss(p_hat, t_hat, mask)

        total = (
            self.alpha["ssi"] * ssi_loss
            + self.alpha["hdn"] * hdn_loss
            + self.alpha["grad"] * grad_loss
            + self.alpha["normal"] * normal_loss
        )

        ssi_v, hdn_v, grad_v, norm_v = torch.stack([
            ssi_loss.detach(),
            hdn_loss.detach(),
            grad_loss.detach(),
            normal_loss.detach(),
        ]).tolist()
        return total, {
            "ssi": ssi_v,
            "hdn": hdn_v,
            "grad": grad_v,
            "normal": norm_v,
        }

    # ------------------------------------------------------------------
    def _normalize_global(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dense_mode: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target = self._squeeze(pred, target)
        if dense_mode is None:
            dense_mode = mask is None
        if not dense_mode and mask is not None:
            (mask,) = self._squeeze(mask)
        eff_mask = None if dense_mode else mask
        return normalize_depth(pred, target, mask=eff_mask, mad_denom_eps=0.0)

    def _trimmed_ssi_loss(
        self,
        p_hat: torch.Tensor,
        t_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dense_mode: Optional[bool] = None,
    ) -> torch.Tensor:
        p_hat, t_hat = self._squeeze(p_hat, t_hat)
        if dense_mode is None:
            dense_mode = mask is None
        if not dense_mode and mask is not None:
            (mask,) = self._squeeze(mask)
        eff_mask = None if dense_mode else mask
        return trimmed_ssi_loss(p_hat, t_hat, mask=eff_mask, trim_ratio=self.trim_ratio)

    def _hybrid_depth_norm_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Distill Any Depth (arXiv:2502.19204) / HDN (NeurIPS 2022) Shared-Context Normalization.
        Partitions the teacher's depth map `target` into S in {2, 4} depth segments and normalizes
        pred and target independently within each depth segment.
        """
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            (mask,) = self._squeeze(mask)
        return fast_hdn_loss(
            pred,
            target,
            mask=mask,
            segments=self.hdn_segments,
            mode=self.hdn_mode,
        )

    def _gradient_and_normal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            (mask,) = self._squeeze(mask)
        return grad_normal_loss(
            pred,
            target,
            mask=mask,
            grad_scales=self.grad_scales,
            normal_beta=self.normal_beta,
            compute_normal=(self.alpha["normal"] > 0),
        )

    @staticmethod
    def _squeeze(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)

