from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_classification_metrics, compute_confusion_matrix, compute_per_class_metrics


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
    test_metrics["test_loss"] = test_loss
    class_metrics = compute_per_class_metrics(test_true, test_pred, num_classes)

    cm = compute_confusion_matrix(test_true, test_pred)
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
