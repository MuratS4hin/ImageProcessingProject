from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    data_root: str = "/content/data"
    output_root: str = "/content/outputs"

    model: str = "resnet50"
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
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransformConfig:
    color_space: str = "rgb"
    texture_filter: str = "none"
    texture_strength: float = 0.35


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val, max_val = image.min(), image.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def sobel_edge_map(gray_image_uint8: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray_image_uint8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image_uint8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_to_unit(magnitude)


def gabor_edge_map(gray_image_uint8: np.ndarray) -> np.ndarray:
    gray = gray_image_uint8.astype(np.float32)
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    wavelengths = [3.0, 5.0]

    responses = []
    for theta in orientations:
        for lam in wavelengths:
            kernel = cv2.getGaborKernel(
                ksize=(7, 7),
                sigma=2.0,
                theta=theta,
                lambd=lam,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            responses.append(np.abs(response))

    stacked = np.stack(responses, axis=0)
    max_response = np.max(stacked, axis=0)
    return _normalize_to_unit(max_response)


def _convert_color(rgb_uint8: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return rgb_uint8
    if color_space == "hsv":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    if color_space == "lab":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    raise ValueError(f"Unsupported color space: {color_space}")


def _texture_map(rgb_uint8: np.ndarray, texture_filter: str) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    if texture_filter == "none":
        return np.zeros_like(gray, dtype=np.float32)
    if texture_filter == "sobel":
        return sobel_edge_map(gray)
    if texture_filter == "gabor":
        return gabor_edge_map(gray)
    raise ValueError(f"Unsupported texture filter: {texture_filter}")


class ColorTextureTransform:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb_uint8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        converted = _convert_color(rgb_uint8, self.cfg.color_space).astype(np.float32) / 255.0

        if self.cfg.texture_filter != "none":
            edge = _texture_map(rgb_uint8, self.cfg.texture_filter)
            edge_3ch = np.stack([edge, edge, edge], axis=-1)
            alpha = float(np.clip(self.cfg.texture_strength, 0.0, 1.0))
            converted = (1.0 - alpha) * converted + alpha * edge_3ch

        converted = np.clip(converted, 0.0, 1.0)
        tensor = torch.from_numpy(converted).permute(2, 0, 1).contiguous().float()
        return tensor


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


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        y_true_onehot = label_binarize(y_true, classes=np.arange(num_classes))
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
        )
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    labels = np.arange(num_classes)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    class_metrics: Dict[str, Dict[str, float]] = {}
    for idx in range(num_classes):
        class_metrics[f"class_{idx}"] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }

    return class_metrics


def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _epoch_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        y_true_all.append(targets.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())

    epoch_loss = float(np.mean(losses)) if losses else 0.0
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)
    return epoch_loss, y_true, y_pred, y_prob


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    run_dir: Path,
    save_best_only: bool = True,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_model_path = run_dir / "best_model.pt"

    epoch_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred, train_prob = _epoch_step(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_true, val_pred, val_prob = _epoch_step(model, val_loader, criterion, None, device)

        train_metrics = compute_classification_metrics(train_true, train_pred, train_prob, num_classes)
        val_metrics = compute_classification_metrics(val_true, val_pred, val_prob, num_classes)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, val_acc={val_metrics['accuracy']:.4f}"
        )

        epoch_history["train_loss"].append(float(train_loss))
        epoch_history["val_loss"].append(float(val_loss))
        epoch_history["train_acc"].append(float(train_metrics["accuracy"]))
        epoch_history["val_acc"].append(float(val_metrics["accuracy"]))

        improved = val_metrics["accuracy"] > (best_val_acc + early_stopping_min_delta)
        should_save = improved
        if should_save:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        elif not save_best_only:
            torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch:03d}.pt")
            epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}: "
                f"no val_acc improvement greater than {early_stopping_min_delta:.6f} "
                f"for {early_stopping_patience} epoch(s)."
            )
            break

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    inference_start = time.perf_counter()
    test_loss, test_true, test_pred, test_prob = _epoch_step(model, test_loader, criterion, None, device)
    inference_time_seconds = float(time.perf_counter() - inference_start)

    test_metrics = compute_classification_metrics(test_true, test_pred, test_prob, num_classes)
    class_metrics = compute_per_class_metrics(test_true, test_pred, num_classes)

    cm = confusion_matrix(test_true, test_pred)
    _save_confusion_matrix(cm, run_dir / "confusion_matrix.png")

    report = {
        "global_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "roc_auc_ovr": float(test_metrics["roc_auc_ovr"]),
            "inference_time_seconds": inference_time_seconds,
            "test_loss": float(test_loss),
            "precision_weighted": float(test_metrics["precision_weighted"]),
            "recall_weighted": float(test_metrics["recall_weighted"]),
            "f1_weighted": float(test_metrics["f1_weighted"]),
        },
        "epoch_history": epoch_history,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    return report


def create_resnet50_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Colab grid runner for ResNet50")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--data-root", default="/content/data")
    parser.add_argument("--output-root", default="/content/outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    ns, _ = parser.parse_known_args()

    color_spaces = ["rgb", "hsv", "lab"]
    texture_filters = ["none", "gabor", "sobel"]

    set_seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = timestamp()
    root_out = ensure_dir(Path(ns.output_root) / f"resnet50_grid_{stamp}")

    summary: list[dict] = []
    total = len(color_spaces) * len(texture_filters)
    run_idx = 0

    for color_space in color_spaces:
        for texture_filter in texture_filters:
            run_idx += 1

            cfg = ExperimentConfig(
                dataset=ns.dataset,
                data_root=ns.data_root,
                output_root=str(root_out),
                model="resnet50",
                color_space=color_space,
                texture_filter=texture_filter,
                epochs=ns.epochs,
                batch_size=ns.batch_size,
                lr=ns.lr,
                weight_decay=ns.weight_decay,
                val_split=ns.val_split,
                num_workers=ns.num_workers,
                seed=ns.seed,
                save_best_only=not ns.save_all,
                early_stopping_patience=ns.early_stopping_patience,
                early_stopping_min_delta=ns.early_stopping_min_delta,
            )

            run_name = f"{cfg.dataset}_resnet50_{cfg.color_space}_{cfg.texture_filter}"
            run_dir = ensure_dir(root_out / run_name)

            print("=" * 80)
            print(f"[{run_idx}/{total}] Running: {run_name} on device={device}")

            train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)
            model = create_resnet50_model(num_classes)
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            training_report = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                epochs=cfg.epochs,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                device=device,
                run_dir=run_dir,
                save_best_only=cfg.save_best_only,
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_min_delta=cfg.early_stopping_min_delta,
            )

            report = {
                "model_config": {
                    "color_space": cfg.color_space.upper(),
                    "filter": cfg.texture_filter.capitalize(),
                    "trainable_parameters": int(trainable_parameters),
                },
                **training_report,
            }

            save_json(cfg.to_dict(), run_dir / "config.json")
            save_json(report, run_dir / "metrics.json")

            summary_item = {
                "run": run_name,
                **report["global_metrics"],
            }
            summary.append(summary_item)
            print(f"Completed {run_name} | metrics={report['global_metrics']}")

    save_json({"results": summary}, root_out / "summary.json")
    print(f"All experiments completed. Summary: {root_out / 'summary.json'}")


if __name__ == "__main__":
    main()
