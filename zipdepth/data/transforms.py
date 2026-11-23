import math
import numpy as np
import contextlib

if not hasattr(np, '_no_nep50_warning'):
    @contextlib.contextmanager
    def _dummy_no_nep50_warning():
        yield
    np._no_nep50_warning = _dummy_no_nep50_warning

import albumentations as A
from typing import Tuple, Optional
import cv2
import torch
cv2.setNumThreads(0)


class AlbumentationsWrapper:
    """Wrapper: transform(image, depth) -> (image, depth)"""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if image.dtype not in [np.uint8, np.float32]:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        if depth is not None:
            if depth.ndim == 3:
                depth = depth.squeeze()
            h, w = image.shape[:2]
            dh, dw = depth.shape[:2]
            if (dh, dw) != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            result = self.transform(image=image, mask=depth)
            return result['image'], result['mask']
        else:
            result = self.transform(image=image)
            return result['image'], None

    def __repr__(self):
        return repr(self.transform)


def get_train_transforms(height: int = 512, width: int = 512, color_jitter_p: float = 0.8) -> AlbumentationsWrapper:
    """
    Training transforms: random crop/resize + horizontal flip + color jitter.
    """
    ops = [
        A.OneOf(
            [
                A.Compose([
                    A.SmallestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LINEAR),
                    A.RandomCrop(height=height, width=width),
                ]),
                A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
    ]
    if color_jitter_p > 0.0:
        ops.append(
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=color_jitter_p,
            )
        )
    ops.extend([
        A.ChannelShuffle(p=0.05),
        A.ToGray(num_output_channels=3, p=0.05),
    ])

    transforms = A.Compose(
        ops,
        additional_targets={'mask': 'mask'},
    )
    return AlbumentationsWrapper(transforms)


def apply_depth_cutmix(images, depth, p=0.5, alpha=1.0):
    """Apply batch-level CutMix consistently to RGB images and depth maps.

    Expects `images` with shape [B, C, H, W] and `depth` with shape [B, 1, H, W]
    or [B, H, W] on the same device.
    """
    batch_size = images.size(0)
    if batch_size < 2 or p <= 0.0 or torch.rand(1).item() >= p:
        return images, depth

    _, _, height, width = images.shape
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0.0 else 1.0
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))

    cx = int(torch.randint(0, width, (1,)).item())
    cy = int(torch.randint(0, height, (1,)).item())
    x1 = max(0, cx - cut_w // 2)
    x2 = min(width, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(height, cy + cut_h // 2)

    if x2 <= x1 or y2 <= y1:
        return images, depth

    perm = torch.randperm(batch_size, device=images.device)
    mixed_images = images.clone()
    mixed_depth = depth.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    if mixed_depth.ndim == 4:
        mixed_depth[:, :, y1:y2, x1:x2] = depth[perm, :, y1:y2, x1:x2]
    else:
        mixed_depth[:, y1:y2, x1:x2] = depth[perm, y1:y2, x1:x2]
    return mixed_images, mixed_depth
