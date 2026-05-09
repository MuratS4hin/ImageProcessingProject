from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    output_root: str = "./outputs"

    model: str = "resnet18"
    color_space: str = "rgb"
    texture_filter: str = "none"

    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42

    save_best_only: bool = True

    @staticmethod
    def from_namespace(ns: argparse.Namespace) -> "ExperimentConfig":
        return ExperimentConfig(
            dataset=ns.dataset,
            data_root=ns.data_root,
            output_root=ns.output_root,
            model=ns.model,
            color_space=ns.color_space,
            texture_filter=ns.texture_filter,
            epochs=ns.epochs,
            batch_size=ns.batch_size,
            lr=ns.lr,
            weight_decay=ns.weight_decay,
            val_split=ns.val_split,
            num_workers=ns.num_workers,
            seed=ns.seed,
            save_best_only=not ns.save_all,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Color space + texture filtering experiments on CIFAR"
    )

    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./outputs")

    parser.add_argument("--model", choices=["alexnet", "resnet18", "densenet121"], default="resnet18")
    parser.add_argument(
        "--color-space",
        choices=["rgb", "hsv", "lab", "xyz", "ycrcb", "gray"],
        default="rgb",
    )
    parser.add_argument("--texture-filter", choices=["none", "sobel", "gabor"], default="none")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-all", action="store_true")

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["alexnet", "resnet18", "densenet121"],
        help="Optional multi-run override for model list",
    )
    parser.add_argument(
        "--color-spaces",
        nargs="+",
        choices=["rgb", "hsv", "lab", "xyz", "ycrcb", "gray"],
        help="Optional multi-run override for color space list",
    )
    parser.add_argument(
        "--texture-filters",
        nargs="+",
        choices=["none", "sobel", "gabor"],
        help="Optional multi-run override for texture filter list",
    )

    return parser


def expand_runs(base_cfg: ExperimentConfig, ns: argparse.Namespace) -> List[ExperimentConfig]:
    models = ns.models if ns.models else [base_cfg.model]
    color_spaces = ns.color_spaces if ns.color_spaces else [base_cfg.color_space]
    texture_filters = ns.texture_filters if ns.texture_filters else [base_cfg.texture_filter]

    expanded: List[ExperimentConfig] = []
    for model in models:
        for color_space in color_spaces:
            for texture_filter in texture_filters:
                cfg = ExperimentConfig(**base_cfg.to_dict())
                cfg.model = model
                cfg.color_space = color_space
                cfg.texture_filter = texture_filter
                expanded.append(cfg)

    return expanded


def make_run_name(cfg: ExperimentConfig) -> str:
    return f"{cfg.dataset}_{cfg.model}_{cfg.color_space}_{cfg.texture_filter}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
