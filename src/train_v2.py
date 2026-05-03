"""Train v2: same as train.py but with per-feature L2 normalization
applied to the HOG, HSV, and BOVW blocks separately before concatenation.

This is the 'one small improvement' I proposed in the Project 04 report.
The rationale: HOG (1764 dims), HSV (4096 dims), and BOVW (100 dims) have
very different scales and dimensionalities. Running StandardScaler on the
concatenated vector lets the bigger blocks dominate the SVM's RBF kernel.
Normalizing each block to unit L2 length first puts them on equal footing.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_split, build_codebook, bovw_histogram, print_confusion_matrix


def make_feature_matrix_v2(split_data, codebook, k):
    """Same blocks as v1, but each block is L2-normalized before concatenation."""
    n = len(split_data["labels"])
    bovw = np.zeros((n, k), dtype=np.float32)
    i = 0
    while i < n:
        bovw[i] = bovw_histogram(split_data["orb"][i], codebook, k)
        i += 1
    hog_norm = normalize(split_data["hog"], norm="l2")
    hsv_norm = normalize(split_data["hsv"], norm="l2")
    bovw_norm = normalize(bovw, norm="l2")
    X = np.hstack([hog_norm, hsv_norm, bovw_norm])
    return X, split_data["labels"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features")
    p.add_argument("--models", default="models_v2")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--gamma", default="scale")
    args = p.parse_args()

    os.makedirs(args.models, exist_ok=True)

    print("Loading features...")
    train = load_split(args.features, "train")
    val = load_split(args.features, "val")
    test = load_split(args.features, "test")
    classes = list(train["classes"])

    print("\nBuilding BOVW codebook...")
    codebook = build_codebook(train["orb"], k=args.k)

    print("\nBuilding L2-normalized feature matrices...")
    X_train, y_train = make_feature_matrix_v2(train, codebook, args.k)
    X_val, y_val = make_feature_matrix_v2(val, codebook, args.k)
    X_test, y_test = make_feature_matrix_v2(test, codebook, args.k)

    print("\nStandardizing...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\nTraining SVM (RBF, C=%s, class_weight=balanced)..." % args.C)
    clf = SVC(kernel="rbf", C=args.C, gamma=args.gamma, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y_train)

    pred_train = clf.predict(X_train_s)
    pred_val = clf.predict(X_val_s)
    pred_test = clf.predict(X_test_s)

    acc_train = accuracy_score(y_train, pred_train)
    acc_val = accuracy_score(y_val, pred_val)
    acc_test = accuracy_score(y_test, pred_test)

    print("\n=== Results (v2: per-block L2 normalization) ===")
    print("Train: %.4f  (%d / %d)" % (acc_train, (pred_train == y_train).sum(), len(y_train)))
    print("Val:   %.4f  (%d / %d)" % (acc_val, (pred_val == y_val).sum(), len(y_val)))
    print("Test:  %.4f  (%d / %d)" % (acc_test, (pred_test == y_test).sum(), len(y_test)))

    print("\nConfusion matrix (test):")
    cm = confusion_matrix(y_test, pred_test)
    print_confusion_matrix(cm, classes)

    print("\nClassification report (test):")
    print(classification_report(y_test, pred_test, target_names=classes))

    # Save artifacts
    f = open(os.path.join(args.models, "svm.pkl"), "wb")
    pickle.dump(clf, f); f.close()
    f = open(os.path.join(args.models, "scaler.pkl"), "wb")
    pickle.dump(scaler, f); f.close()
    f = open(os.path.join(args.models, "codebook.pkl"), "wb")
    pickle.dump(codebook, f); f.close()
    f = open(os.path.join(args.models, "classes.pkl"), "wb")
    pickle.dump(classes, f); f.close()

    f = open(os.path.join(args.models, "metrics.txt"), "w")
    f.write("v2 = per-block L2 normalization before StandardScaler\n")
    f.write("train_accuracy: %.4f\n" % acc_train)
    f.write("val_accuracy:   %.4f\n" % acc_val)
    f.write("test_accuracy:  %.4f\n" % acc_test)
    f.write("\nConfusion matrix (test):\n%s\n" % str(cm))
    f.write(classification_report(y_test, pred_test, target_names=classes))
    f.close()
    print("\ndone.")


if __name__ == "__main__":
    main()
