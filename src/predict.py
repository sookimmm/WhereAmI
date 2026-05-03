"""Predict the building class for a single image.

Loads the trained SVM, scaler, BOVW codebook, and class list, runs the same
preprocessing/segmentation/feature pipeline used during training, then prints
the predicted class and per-class probabilities.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess
from segment import grabcut_building
from features import hog_descriptor, hsv_histogram, orb_features
from train import bovw_histogram


def load_pickle(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to a JPG/PNG photo")
    p.add_argument("--models", default="models")
    args = p.parse_args()

    # 1. load saved model artifacts
    clf = load_pickle(os.path.join(args.models, "svm.pkl"))
    scaler = load_pickle(os.path.join(args.models, "scaler.pkl"))
    codebook = load_pickle(os.path.join(args.models, "codebook.pkl"))
    classes = load_pickle(os.path.join(args.models, "classes.pkl"))
    k = codebook.n_clusters

    # 2. load and preprocess the input image (same settings as training)
    img = cv2.imread(args.image)
    if img is None:
        print("ERROR: could not read image:", args.image)
        sys.exit(1)
    img = preprocess(img, max_side=400)
    mask, _ = grabcut_building(img, iters=2)

    # 3. extract the same three features used during training
    hog_vec = hog_descriptor(img)
    hsv_vec = hsv_histogram(img, mask)
    _, orb_des = orb_features(img)
    bovw_vec = bovw_histogram(orb_des, codebook, k)

    # 4. concatenate, standardize, predict
    feat = np.hstack([hog_vec, hsv_vec, bovw_vec]).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred = int(clf.predict(feat_s)[0])
    scores = clf.decision_function(feat_s)[0]

    # 5. print result
    print("Image     :", args.image)
    print("Prediction:", classes[pred])
    print("\nDecision scores (higher = more confident):")
    i = 0
    while i < len(classes):
        print("  %-12s %+.4f" % (classes[i], scores[i]))
        i += 1


if __name__ == "__main__":
    main()
