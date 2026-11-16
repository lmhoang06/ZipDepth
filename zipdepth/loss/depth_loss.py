"""Scale-and-shift invariant depth loss with multi-scale gradient matching."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from zipdepth.loss._triton_ops import (
    grad_normal_loss,
    normalize_depth,
    ssi_loss as fast_ssi_loss,
)


class ZipDepthLoss(nn.Module):
    """
    Combined loss for affine-invariant monocular depth estimation.

    Normalizes predictions and targets via median/MAD before computing:
      - SSI loss:      masked L1 on normalized maps
      - Gradient loss: masked L1 on image gradients at multiple scales
    """

    def __init__(
        self,
        alpha_ssi: float = 1.0,
        alpha_grad: float = 2.0,
        grad_scales: int = 4,
    ) -> None:
        super().__init__()
        self.alpha = {'ssi': alpha_ssi, 'grad': alpha_grad}
        self.grad_scales = grad_scales

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Training uses DENSE proxy depth (Depth Anything V2) when mask is None;
        # avoid materializing an all-ones mask and route directly to dense kernels.
        # For SPARSE ground truth, pass an explicit mask=(target > 0).float().
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            mask = self._squeeze(mask)[0]

        p_hat, t_hat = normalize_depth(pred, target, mask=mask, mad_denom_eps=1e-8)
        ssi_loss = fast_ssi_loss(p_hat, t_hat, mask=mask)
        grad_loss, _ = grad_normal_loss(
            p_hat,
            t_hat,
            mask=mask,
            grad_scales=self.grad_scales,
            compute_normal=False,
        )

        total = self.alpha['ssi'] * ssi_loss + self.alpha['grad'] * grad_loss
        ssi_val, grad_val = torch.stack([ssi_loss.detach(), grad_loss.detach()]).tolist()
        return total, {'ssi': ssi_val, 'grad': grad_val}

    # ------------------------------------------------------------------
    def _normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            mask = self._squeeze(mask)[0]
        return normalize_depth(pred, target, mask=mask, mad_denom_eps=1e-8)

    def _ssi_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            mask = self._squeeze(mask)[0]
        return fast_ssi_loss(pred, target, mask=mask)

    def _gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, target = self._squeeze(pred, target)
        if mask is not None:
            mask = self._squeeze(mask)[0]
        grad_loss, _ = grad_normal_loss(
            pred,
            target,
            mask=mask,
            grad_scales=self.grad_scales,
            compute_normal=False,
        )
        return grad_loss

    @staticmethod
    def _squeeze(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(t.squeeze(1) if t.dim() == 4 else t for t in tensors)
