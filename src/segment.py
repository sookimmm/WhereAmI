"""Segmentation: isolate the building region from sky, ground, and clutter."""
import cv2
import numpy as np


def grabcut_building(img, inset=0.08, iters=3):
    """
    GrabCut with an inset rectangle as the 'probable foreground' prior.
    Buildings fill the central ~85% of every shot in this dataset, while sky,
    ground, and bystanders sit near the borders -- so an inset rect is a
    reasonable shape-free prior.
    Returns (mask_uint8, masked_img). mask is 0 or 255.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    x0, y0 = int(w * inset), int(h * inset)
    x1, y1 = int(w * (1 - inset)), int(h * (1 - inset))
    rect = (x0, y0, x1 - x0, y1 - y0)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Clean small specks and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    masked = cv2.bitwise_and(img, img, mask=fg)
    return fg, masked
