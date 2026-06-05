# The Impact of Color Space Transformations and Texture Filtering on CNN-Based Image Classification

This project is a starter implementation for your Advanced Topics in Image Analysis course proposal.

## What is implemented

- CIFAR-10 and CIFAR-100 support
- Color spaces: `rgb`, `hsv`, `lab`, `xyz`, `ycrcb`, `gray`
- Texture filtering modes: `none`, `sobel`, `gabor`
- CNN architectures: `alexnet`, `resnet18`, `densenet121`
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC (OvR)
- Reproducible experiment outputs (JSON + confusion matrix image)

## Project structure

```
.
├── main.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── filters.py
    ├── metrics.py
    ├── models.py
    ├── train.py
    ├── transforms.py
    └── utils.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example runs

Single experiment:

```bash
python main.py \
  --dataset cifar10 \
  --model resnet18 \
  --color-space lab \
  --texture-filter sobel \
  --epochs 20 \
  --batch-size 128
```

Grid experiment (all combinations):

```bash
python main.py \
  --dataset cifar10 \
  --models alexnet resnet18 densenet121 \
  --color-spaces rgb hsv lab xyz ycrcb gray \
  --texture-filters none sobel gabor \
  --epochs 30 \
  --batch-size 128
```
``` For Windows
python main.py --dataset cifar10 --models resnet18 densenet121 --color-spaces rgb hsv lab gray --texture-filters none sobel gabor --epochs 30 --batch-size 128 --early-stopping-patience 3 --early-stopping-min-delta 0.001

python main.py --dataset cifar10 --models alexnet resnet18 --color-spaces rgb hsv lab --texture-filters none sobel gabor --epoch 50 --early-stopping-patience 3 --early-stopping-min-delta 0.001

python main.py --model simplecnn --color-spaces rgb hsv lab --texture-filters none sobel gabor --epochs 30 --batch-size 128 --lr 0.001 --early-stopping-patience 3 --early-stopping-min-delta 0.001
```

## Output

Each run writes to `outputs/<timestamp>/<run_name>/`:

- `metrics.json`
- `config.json`
- `confusion_matrix.png`
- `best_model.pt`

## Notes

- `gray` is replicated to 3 channels so all backbones can be used consistently.
- ROC-AUC is computed as one-vs-rest using predicted probabilities.
