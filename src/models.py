from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.features[2] = nn.MaxPool2d(kernel_size=2, stride=2)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")
