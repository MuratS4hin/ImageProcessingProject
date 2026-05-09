from __future__ import annotations

import cv2
import numpy as np


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
