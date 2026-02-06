"""Data transforms for HLS Burn Scars segmentation."""

import albumentations
import albumentations.pytorch


def get_train_transforms() -> list:
    """Get default training transforms.

    Returns:
        List of albumentations transforms for training.
    """
    return [
        albumentations.D4(),  # Random flips and 90° rotations (D4 symmetry group)
        albumentations.pytorch.transforms.ToTensorV2(),
    ]


def get_val_transforms() -> list:
    """Get default validation transforms.

    Returns:
        List of albumentations transforms for validation.
    """
    return [
        albumentations.pytorch.transforms.ToTensorV2(),
    ]


def get_test_transforms() -> list:
    """Get default test transforms.

    Returns:
        List of albumentations transforms for testing.
    """
    return [
        albumentations.pytorch.transforms.ToTensorV2(),
    ]


def get_weak_transforms() -> list:
    """Get weak augmentation transforms for SSL teacher input.

    Same as get_train_transforms() — D4 geometric augmentation only.
    Kept as a separate function for semantic clarity in SSL pipelines.

    Returns:
        List of albumentations transforms (D4 + ToTensorV2).
    """
    return get_train_transforms()


def get_strong_transforms(n_channels: int = 6) -> list:
    """Get strong augmentation transforms for SSL student input.

    Applies aggressive photometric + geometric augmentations designed
    to create a challenging view that the student must learn to match
    against teacher pseudo-labels from the weakly-augmented view.

    Pipeline:
        1. D4 (flips + 90-degree rotations)
        2. RandomBrightnessContrast (per-channel, +/-20%)
        3. GaussNoise (std 0.03–0.1)
        4. GaussianBlur (kernel 3–7)
        5. ChannelDropout (drop 1 of n_channels bands, p=0.3)
        6. ToTensorV2

    Args:
        n_channels: Number of input channels. Defaults to 6 (HLS bands).

    Returns:
        List of albumentations transforms for strong augmentation.
    """
    return [
        albumentations.D4(),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        albumentations.GaussNoise(std_range=(0.03, 0.1), p=0.5),
        albumentations.GaussianBlur(blur_limit=(3, 7), p=0.5),
        albumentations.ChannelDropout(
            channel_drop_range=(1, 1),
            p=0.3,
        ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]


def get_augmented_train_transforms() -> list:
    """Get more aggressive training transforms for regularization.

    Includes additional augmentations beyond basic flips/rotations.

    Returns:
        List of albumentations transforms for training with more augmentation.
    """
    return [
        albumentations.D4(),
        albumentations.OneOf(
            [
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5,
                ),
                albumentations.GaussNoise(std_range=(0.03, 0.1), p=0.5),
            ],
            p=0.3,
        ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
