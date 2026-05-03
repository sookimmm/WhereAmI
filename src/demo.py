"""End-to-end demo: run preprocess -> segment -> features on one image
and save a visualization grid to outputs/."""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from segment import grabcut_building
from features import canny_edges, hough_lines, orb_features


def draw_lines(img, lines):
    out = img.copy()
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def draw_keypoints(img, kps):
    return cv2.drawKeypoints(img, kps, None, color=(0, 255, 0),
                             flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(image_path, out_path):
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(image_path)

    pre = preprocess(raw)
    mask, masked = grabcut_building(pre)

    edges = canny_edges(pre)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    lines = hough_lines(edges_masked)
    kps, _ = orb_features(pre)

    vis_lines = draw_lines(pre, lines)
    vis_kps = draw_keypoints(pre, kps)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (bgr2rgb(cv2.resize(raw, (pre.shape[1], pre.shape[0]))), "1. Original (resized)"),
        (bgr2rgb(pre), "2. Preprocessed (denoise + CLAHE)"),
        (bgr2rgb(masked), "3. Segmentation (GrabCut)"),
        (edges_masked, "4. Canny edges (masked)"),
        (bgr2rgb(vis_lines), f"5. Hough lines ({len(lines)})"),
        (bgr2rgb(vis_kps), f"6. ORB keypoints ({len(kps)})"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        cmap = "gray" if im.ndim == 2 else None
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"WhereAmI pipeline: {os.path.basename(image_path)}", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to a sample JPG")
    p.add_argument("--out", default="outputs/demo.png", help="output PNG path")
    a = p.parse_args()
    run(a.image, a.out)
