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
