"""
Augmentation Configuration

The augmentation pipeline that is specified in the augmentation object
will be exported in the training script
"""
from typing import Tuple
import numpy as np
import albumentations as A


def identity(image: np.ndarray, mask: np.ndarray):
    """Identity Function.
    Use if no preprocessing or augmentation is required
    """
    return {"image": image, "mask": mask}


def _preprocessing():
    transform = A.Compose([A.LongestMaxSize(max_size=1024, p=1)])
    return transform


def _augmentation(patch_dim: Tuple[int, int] = (512, 512)):
    transform = A.Compose(
        [
            # This allows meaningful yet stochastic cropped views
            A.CropNonEmptyMaskIfExists(patch_dim[0], patch_dim[1], p=1),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.25),
            A.ColorJitter(p=0.25),
            A.GaussNoise(p=0.25),
            A.CoarseDropout(p=0.5, max_holes=64, max_height=8, max_width=8),
            A.RandomBrightnessContrast(p=0.25),
        ],
    )
    return transform


preprocessing = _preprocessing()
augmentation = _augmentation()
