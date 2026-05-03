"""Project 05: generate visualizations for the report:
- outputs/test_confusion_matrix.png : confusion matrix on the test set
- outputs/test_failures.png         : grid of misclassified test images
"""
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_split, bovw_histogram


def load_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def make_v2_features(split_data, codebook, k):
    n = len(split_data["labels"])
    bovw = np.zeros((n, k), dtype=np.float32)
    i = 0
    while i < n:
        bovw[i] = bovw_histogram(split_data["orb"][i], codebook, k)
        i += 1
    hog_n = normalize(split_data["hog"], norm="l2")
    hsv_n = normalize(split_data["hsv"], norm="l2")
    bovw_n = normalize(bovw, norm="l2")
    X = np.hstack([hog_n, hsv_n, bovw_n])
    return X


def main():
    clf = load_pickle("models_v2/svm.pkl")
    scaler = load_pickle("models_v2/scaler.pkl")
    codebook = load_pickle("models_v2/codebook.pkl")
    classes = load_pickle("models_v2/classes.pkl")
    k = codebook.n_clusters

    test = load_split("data/features", "test")
    raw = np.load("data/features/test.npz", allow_pickle=True)
    paths = list(raw["paths"])

    X_test = make_v2_features(test, codebook, k)
    X_test_s = scaler.transform(X_test)
    pred = clf.predict(X_test_s)
    y_test = test["labels"]
    cm = confusion_matrix(y_test, pred)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n_correct = (pred == y_test).sum()
    ax.set_title("Test Confusion Matrix\n(%d/%d = %.1f%% accuracy)" % (n_correct, len(y_test), 100 * n_correct / len(y_test)))
    i = 0
    while i < len(classes):
        j = 0
        while j < len(classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
            j += 1
        i += 1
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/test_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("saved outputs/test_confusion_matrix.png")

    # Misclassified images grid (up to 6)
    wrong_idx = []
    i = 0
    while i < len(pred):
        if pred[i] != y_test[i]:
            wrong_idx.append(i)
        i += 1
    print("misclassified test images:", len(wrong_idx))

    n_show = min(6, len(wrong_idx))
    if n_show == 0:
        return
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    i = 0
    while i < n_show:
        idx = wrong_idx[i]
        img = cv2.imread(paths[idx])
        if img is None:
            i += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = 600 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ax = flat[i]
        ax.imshow(img)
        ax.set_title("True: %s\nPred: %s" % (classes[y_test[idx]], classes[pred[idx]]),
                    color="red", fontsize=11)
        ax.axis("off")
        i += 1
    while i < len(flat):
        flat[i].axis("off")
        i += 1
    fig.suptitle("Misclassified test images", fontsize=14)
    fig.tight_layout()
    fig.savefig("outputs/test_failures.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved outputs/test_failures.png")


if __name__ == "__main__":
    main()
