from zipdepth.loss.depth_loss import ZipDepthLoss
from zipdepth.loss.hdn_loss import HDNDistillationLoss


def create_loss(loss_type="original", **kwargs):
    """
    Factory to instantiate ZipDepth loss functions.

    Args:
        loss_type: 'original' (standard ZipDepthLoss: Global SSI + Multi-scale gradient)
                   or 'hdn' / 'upgraded' (HDNDistillationLoss: Trimmed Global SSI + HDN-DP + Normal)
        **kwargs: Arguments forwarded to the selected loss class.

    Returns:
        nn.Module loss instance
    """
    name = str(loss_type).lower().strip()
    if name in ("original", "base", "zipdepth", "default"):
        allowed = ("alpha_ssi", "alpha_grad", "grad_scales")
        loss_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return ZipDepthLoss(**loss_kwargs)
    elif name in ("hdn", "upgraded", "distill_any_depth"):
        return HDNDistillationLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Supported choices: 'original', 'hdn'."
        )


__all__ = ["ZipDepthLoss", "HDNDistillationLoss", "create_loss"]
