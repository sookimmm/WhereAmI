"""Project 05: evaluate the final model on the held-out test set.

The test split was created by src/split.py at the start of the project and
has not been touched during training, validation, or hyperparameter tuning.
That makes it 'unknown data' for the purpose of the final evaluation.
"""
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_split, make_feature_matrix, print_confusion_matrix


def load_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def main():
    print("Loading model artifacts...")
    clf = load_pickle("models/svm.pkl")
    scaler = load_pickle("models/scaler.pkl")
    codebook = load_pickle("models/codebook.pkl")
    classes = load_pickle("models/classes.pkl")
    k = codebook.n_clusters

    print("Loading test features...")
    test = load_split("data/features", "test")
    X_test, y_test = make_feature_matrix(test, codebook, k)
    X_test_s = scaler.transform(X_test)

    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    n_correct = (pred == y_test).sum()

    print("\n=== Test Set Results ===")
    print("Test accuracy: %.4f  (%d / %d correct)" % (acc, n_correct, len(y_test)))

    print("\nConfusion matrix (test):")
    cm = confusion_matrix(y_test, pred)
    print_confusion_matrix(cm, classes)

    print("\nClassification report (test):")
    print(classification_report(y_test, pred, target_names=classes))

    # Save metrics + list of misclassified images
    os.makedirs("outputs", exist_ok=True)
    f = open("outputs/test_metrics.txt", "w")
    f.write("test_accuracy: %.4f\n" % acc)
    f.write("correct: %d / %d\n\n" % (n_correct, len(y_test)))
    f.write("Confusion matrix:\n%s\n\n" % str(cm))
    f.write(classification_report(y_test, pred, target_names=classes))
    f.write("\n\nMisclassified test images:\n")
    paths = test.get("paths") if hasattr(test, "get") else None
    # paths array isn't loaded by load_split; reload from npz directly
    raw = np.load("data/features/test.npz", allow_pickle=True)
    paths = list(raw["paths"])
    i = 0
    while i < len(pred):
        if pred[i] != y_test[i]:
            f.write("  %s : true=%s pred=%s\n" % (paths[i], classes[y_test[i]], classes[pred[i]]))
        i += 1
    f.close()
    print("\nWrote outputs/test_metrics.txt")


if __name__ == "__main__":
    main()
