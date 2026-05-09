from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_classification_metrics, compute_confusion_matrix


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
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_model_path = run_dir / "best_model.pt"

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

        should_save = val_metrics["accuracy"] > best_val_acc
        if should_save:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
        elif not save_best_only:
            torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch:03d}.pt")

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_true, test_pred, test_prob = _epoch_step(model, test_loader, criterion, None, device)
    test_metrics = compute_classification_metrics(test_true, test_pred, test_prob, num_classes)
    test_metrics["test_loss"] = test_loss

    cm = compute_confusion_matrix(test_true, test_pred)
    _save_confusion_matrix(cm, run_dir / "confusion_matrix.png")

    return test_metrics
