from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100

from .config import ExperimentConfig
from .transforms import ColorTextureTransform, TransformConfig


def _get_dataset_class(name: str):
    if name == "cifar10":
        return CIFAR10, 10
    if name == "cifar100":
        return CIFAR100, 100
    raise ValueError(f"Unsupported dataset: {name}")


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset_cls, num_classes = _get_dataset_class(cfg.dataset)

    transform = ColorTextureTransform(
        TransformConfig(
            color_space=cfg.color_space,
            texture_filter=cfg.texture_filter,
            texture_strength=0.35,
        )
    )

    full_train = dataset_cls(root=cfg.data_root, train=True, download=True, transform=transform)
    test_set = dataset_cls(root=cfg.data_root, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * cfg.val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes
