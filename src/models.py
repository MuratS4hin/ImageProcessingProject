from __future__ import annotations

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple CNN with ReLU activations for CIFAR classification."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.features[2] = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Modify classifier with dropout - keep original structure
        original_classifier = model.classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            original_classifier[1],  # Linear(9216, 4096)
            original_classifier[2],  # ReLU
            nn.Dropout(p=0.5),
            original_classifier[4],  # Linear(4096, 4096)
            original_classifier[5],  # ReLU
            original_classifier[6],  # Linear(4096, 1000)
        )
        # Replace final layer for num_classes
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
