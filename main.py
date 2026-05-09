from __future__ import annotations

from pathlib import Path

import torch

from src.config import (
    ExperimentConfig,
    build_arg_parser,
    ensure_dir,
    expand_runs,
    make_run_name,
)
from src.data import build_dataloaders
from src.models import create_model
from src.train import run_training
from src.utils import save_json, set_seed, timestamp


def main() -> None:
    parser = build_arg_parser()
    ns = parser.parse_args()

    base_cfg = ExperimentConfig.from_namespace(ns)
    run_cfgs = expand_runs(base_cfg, ns)

    set_seed(base_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = timestamp()
    root_out = ensure_dir(Path(base_cfg.output_root) / stamp)

    summary = []
    for idx, cfg in enumerate(run_cfgs, start=1):
        run_name = make_run_name(cfg)
        run_dir = ensure_dir(root_out / run_name)

        print("=" * 80)
        print(f"[{idx}/{len(run_cfgs)}] Running: {run_name} on device={device}")

        train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)
        model = create_model(cfg.model, num_classes)

        metrics = run_training(
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
        )

        save_json(cfg.to_dict(), run_dir / "config.json")
        save_json(metrics, run_dir / "metrics.json")

        summary_item = {"run": run_name, **metrics}
        summary.append(summary_item)
        print(f"Completed {run_name} | metrics={metrics}")

    save_json({"results": summary}, root_out / "summary.json")
    print(f"All experiments completed. Summary: {root_out / 'summary.json'}")


if __name__ == "__main__":
    main()
