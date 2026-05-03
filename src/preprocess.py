"""Preprocessing: resize, denoise, contrast normalization."""
import cv2
import numpy as np


def resize_max_side(img, max_side=800):
    """Resize so the longest edge == max_side, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def denoise(img):
    """Edge-preserving smoothing. Bilateral keeps edges crisp for Canny later."""
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)


def clahe_lab(img):
    """CLAHE on L channel of LAB. Normalizes lighting without distorting color."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(img, max_side=800):
    """Full preprocessing pipeline."""
    img = resize_max_side(img, max_side)
    img = denoise(img)
    img = clahe_lab(img)
    return img
