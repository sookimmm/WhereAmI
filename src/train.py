"""Train an SVM classifier on the extracted features.

Pipeline:
1. Load HOG, HSV, ORB features for each split.
2. Build a bag-of-visual-words (BOVW) codebook from training ORB descriptors
   using KMeans, then turn each image's ORB descriptors into a fixed-length
   word histogram.
3. Concatenate HOG + HSV + BOVW for each image -> one feature vector.
4. Standardize the feature vector with StandardScaler.
5. Train SVM with RBF kernel.
6. Print train and val accuracy + confusion matrix + classification report.
7. Save the trained model, scaler, and KMeans codebook to models/.
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_split(features_dir, split):
    npz_path = os.path.join(features_dir, split + ".npz")
    pkl_path = os.path.join(features_dir, split + "_orb.pkl")
    data = np.load(npz_path, allow_pickle=True)
    pkl_file = open(pkl_path, "rb")
    orb = pickle.load(pkl_file)
    pkl_file.close()
    return {
        "hog": data["hog"],
        "hsv": data["hsv"],
        "labels": data["labels"],
        "classes": data["classes"],
        "orb": orb,
    }


def build_codebook(orb_list, k=100, seed=42):
    """Stack all training ORB descriptors and cluster them into k visual words."""
    all_des = []
    i = 0
    while i < len(orb_list):
        des = orb_list[i]
        if des is not None and len(des) > 0:
            all_des.append(des)
        i += 1
    stacked = np.vstack(all_des).astype(np.float32)
    print("  total ORB descriptors:", stacked.shape[0])
    print("  clustering into", k, "visual words...")
    km = KMeans(n_clusters=k, n_init=5, random_state=seed)
    km.fit(stacked)
    return km


def bovw_histogram(des, codebook, k):
    """Turn one image's ORB descriptors into a normalized word-count histogram."""
    hist = np.zeros(k, dtype=np.float32)
    if des is None or len(des) == 0:
        return hist
    words = codebook.predict(des.astype(np.float32))
    j = 0
    while j < len(words):
        hist[words[j]] += 1
        j += 1
    s = hist.sum()
    if s > 0:
        hist = hist / s
    return hist


def make_feature_matrix(split_data, codebook, k):
    """Concatenate HOG + HSV + BOVW for every image in a split."""
    n = len(split_data["labels"])
    bovw = np.zeros((n, k), dtype=np.float32)
    i = 0
    while i < n:
        bovw[i] = bovw_histogram(split_data["orb"][i], codebook, k)
        i += 1
    X = np.hstack([split_data["hog"], split_data["hsv"], bovw])
    return X, split_data["labels"]


def print_confusion_matrix(cm, classes):
    header = "           "
    i = 0
    while i < len(classes):
        header = header + ("%-12s" % classes[i])
        i += 1
    print(header)
    i = 0
    while i < len(classes):
        row = "%-10s " % classes[i]
        j = 0
        while j < len(classes):
            row = row + ("%-12d" % cm[i, j])
            j += 1
        print(row)
        i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features")
    p.add_argument("--models", default="models")
    p.add_argument("--k", type=int, default=100, help="BOVW codebook size")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--gamma", default="scale")
    args = p.parse_args()

    os.makedirs(args.models, exist_ok=True)

    print("Loading features...")
    train = load_split(args.features, "train")
    val = load_split(args.features, "val")
    classes = list(train["classes"])
    print("  classes:", classes)
    print("  train n:", len(train["labels"]))
    print("  val   n:", len(val["labels"]))

    print("\nBuilding BOVW codebook from training ORB descriptors...")
    codebook = build_codebook(train["orb"], k=args.k)

    print("\nBuilding feature matrices...")
    X_train, y_train = make_feature_matrix(train, codebook, args.k)
    X_val, y_val = make_feature_matrix(val, codebook, args.k)
    print("  X_train:", X_train.shape)
    print("  X_val  :", X_val.shape)

    print("\nStandardizing...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("\nTraining SVM (RBF kernel, C=%s, gamma=%s)..." % (args.C, args.gamma))
    clf = SVC(kernel="rbf", C=args.C, gamma=args.gamma, random_state=42)
    clf.fit(X_train_s, y_train)

    pred_train = clf.predict(X_train_s)
    pred_val = clf.predict(X_val_s)
    acc_train = accuracy_score(y_train, pred_train)
    acc_val = accuracy_score(y_val, pred_val)

    n_train_correct = (pred_train == y_train).sum()
    n_val_correct = (pred_val == y_val).sum()

    print("\n=== Results ===")
    print("Train accuracy: %.4f  (%d / %d correct)" % (acc_train, n_train_correct, len(y_train)))
    print("Val   accuracy: %.4f  (%d / %d correct)" % (acc_val, n_val_correct, len(y_val)))

    print("\nConfusion matrix (val):")
    cm = confusion_matrix(y_val, pred_val)
    print_confusion_matrix(cm, classes)

    print("\nClassification report (val):")
    print(classification_report(y_val, pred_val, target_names=classes))

    print("Saving model artifacts to", args.models)
    f = open(os.path.join(args.models, "svm.pkl"), "wb")
    pickle.dump(clf, f)
    f.close()
    f = open(os.path.join(args.models, "scaler.pkl"), "wb")
    pickle.dump(scaler, f)
    f.close()
    f = open(os.path.join(args.models, "codebook.pkl"), "wb")
    pickle.dump(codebook, f)
    f.close()
    f = open(os.path.join(args.models, "classes.pkl"), "wb")
    pickle.dump(classes, f)
    f.close()

    metrics_file = open(os.path.join(args.models, "metrics.txt"), "w")
    metrics_file.write("train_accuracy: %.4f\n" % acc_train)
    metrics_file.write("val_accuracy:   %.4f\n" % acc_val)
    metrics_file.write("classes: %s\n" % classes)
    metrics_file.write("\nConfusion matrix (val):\n")
    metrics_file.write(str(cm) + "\n")
    metrics_file.write("\n")
    metrics_file.write(classification_report(y_val, pred_val, target_names=classes))
    metrics_file.close()
    print("done.")


if __name__ == "__main__":
    main()
