from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .filters import gabor_edge_map, sobel_edge_map


@dataclass
class TransformConfig:
    color_space: str = "rgb"
    texture_filter: str = "none"
    texture_strength: float = 0.35


def _convert_color(rgb_uint8: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "rgb":
        return rgb_uint8
    if color_space == "hsv":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    if color_space == "lab":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    if color_space == "xyz":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2XYZ)
    if color_space == "ycrcb":
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb)
    if color_space == "gray":
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        return np.stack([gray, gray, gray], axis=-1)
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
