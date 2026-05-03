"""Project 05: pick one random sample from the test set, run the full pipeline,
and present the result.

Outputs a single PNG with: original image, preprocessed image, segmentation
mask, Canny edges, Hough lines, ORB keypoints, and the predicted class with
decision scores.
"""
import argparse
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from segment import grabcut_building
from features import canny_edges, hough_lines, orb_features, hog_descriptor, hsv_histogram
from train import bovw_histogram


def load_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def list_test_images(test_dir):
    """Return a flat list of (path, true_class) for every image in the test split."""
    samples = []
    classes = sorted(os.listdir(test_dir))
    i = 0
    while i < len(classes):
        cls = classes[i]
        cls_dir = os.path.join(test_dir, cls)
        if os.path.isdir(cls_dir):
            files = sorted(os.listdir(cls_dir))
            j = 0
            while j < len(files):
                f = files[j]
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(cls_dir, f), cls))
                j += 1
        i += 1
    return samples


def draw_lines(img, lines):
    out = img.copy()
    i = 0
    while i < len(lines):
        x1, y1, x2, y2 = lines[i][0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        i += 1
    return out


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default=None,
                   help="directory with test/<class>/*.jpg. Defaults to "
                        "data/splits/test (full split) if it exists, otherwise "
                        "test_samples/ (bundled with the repo).")
    p.add_argument("--models", default="models_v2")
    p.add_argument("--out", default="outputs/random_test_result.png")
    p.add_argument("--seed", type=int, default=None,
                   help="optional RNG seed; omit for a different sample each run")
    args = p.parse_args()

    # Pick test directory: prefer the full split (if extracted), otherwise the
    # small bundle that's committed to the repo so a grader can run this
    # script without re-running split.py / extract_features.py.
    if args.test_dir is not None:
        test_dir = args.test_dir
    elif os.path.isdir("data/splits/test"):
        test_dir = "data/splits/test"
    else:
        test_dir = "test_samples"
    print("Using test directory:", test_dir)

    if args.seed is not None:
        random.seed(args.seed)

    # 1. Pick a random test image
    samples = list_test_images(test_dir)
    if len(samples) == 0:
        print("ERROR: no test images found at", test_dir)
        sys.exit(1)
    image_path, true_class = random.choice(samples)
    print("Random test image:", image_path)
    print("True class:       ", true_class)

    # 2. Load model artifacts
    clf = load_pickle(os.path.join(args.models, "svm.pkl"))
    scaler = load_pickle(os.path.join(args.models, "scaler.pkl"))
    codebook = load_pickle(os.path.join(args.models, "codebook.pkl"))
    classes = load_pickle(os.path.join(args.models, "classes.pkl"))
    k = codebook.n_clusters

    # 3. Run the full pipeline
    raw = cv2.imread(image_path)
    pre = preprocess(raw, max_side=400)
    mask, masked = grabcut_building(pre, iters=2)

    edges = canny_edges(pre)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    lines = hough_lines(edges_masked)
    kps, orb_des = orb_features(pre)

    # 4. Predict with the trained SVM (uses v2 = per-block L2 + balanced classes)
    from sklearn.preprocessing import normalize
    hog_vec = normalize(hog_descriptor(pre).reshape(1, -1), norm="l2")
    hsv_vec = normalize(hsv_histogram(pre, mask).reshape(1, -1), norm="l2")
    bovw_vec = normalize(bovw_histogram(orb_des, codebook, k).reshape(1, -1), norm="l2")
    feat = np.hstack([hog_vec, hsv_vec, bovw_vec])
    feat_s = scaler.transform(feat)
    pred = int(clf.predict(feat_s)[0])
    scores = clf.decision_function(feat_s)[0]
    pred_class = classes[pred]
    is_correct = (pred_class == true_class)

    print("\nPrediction:       ", pred_class, "(CORRECT)" if is_correct else "(WRONG)")
    print("Decision scores:")
    i = 0
    while i < len(classes):
        print("  %-12s %+.4f" % (classes[i], scores[i]))
        i += 1

    # 5. Save a visualization
    vis_lines = draw_lines(pre, lines)
    vis_kps = cv2.drawKeypoints(pre, kps, None, color=(0, 255, 0),
                                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (bgr2rgb(cv2.resize(raw, (pre.shape[1], pre.shape[0]))), "Original"),
        (bgr2rgb(pre), "Preprocessed"),
        (bgr2rgb(masked), "GrabCut segmentation"),
        (edges_masked, "Canny edges (masked)"),
        (bgr2rgb(vis_lines), "Hough lines (%d)" % len(lines)),
        (bgr2rgb(vis_kps), "ORB keypoints (%d)" % len(kps)),
    ]
    flat_axes = axes.ravel()
    i = 0
    while i < len(panels):
        ax = flat_axes[i]
        im, title = panels[i]
        cmap = "gray" if im.ndim == 2 else None
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        i += 1

    verdict = "CORRECT" if is_correct else "WRONG"
    color = "green" if is_correct else "red"
    title = "Test sample: %s | True: %s | Predicted: %s [%s]" % (
        os.path.basename(image_path), true_class, pred_class, verdict
    )
    fig.suptitle(title, fontsize=13, color=color)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()
