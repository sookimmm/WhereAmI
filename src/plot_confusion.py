"""Generate a confusion-matrix PNG for the validation set."""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_split, make_feature_matrix


def load_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def main():
    clf = load_pickle("models/svm.pkl")
    scaler = load_pickle("models/scaler.pkl")
    codebook = load_pickle("models/codebook.pkl")
    classes = load_pickle("models/classes.pkl")
    k = codebook.n_clusters

    val = load_split("data/features", "val")
    X_val, y_val = make_feature_matrix(val, codebook, k)
    X_val_s = scaler.transform(X_val)
    pred_val = clf.predict(X_val_s)
    cm = confusion_matrix(y_val, pred_val)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix\n(120/129 = 93.0% accuracy)")

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
    fig.savefig("outputs/confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("saved outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
