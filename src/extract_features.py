"""Batch-extract features for every image in a directory tree.

Walks data/splits/<split>/<class>/*.jpg, runs preprocess -> segment -> features,
and saves a .npz per split with HOG vectors, HSV histograms, and labels. ORB
descriptors are kept as a list of ndarrays in a separate .pkl since they vary
in length per image.
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


def process_split(split_dir):
    classes = sorted(os.listdir(split_dir))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    hogs, hists, orbs, labels, paths = [], [], [], [], []
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for i, f in enumerate(files, 1):
            path = os.path.join(cls_dir, f)
            img = cv2.imread(path)
            if img is None:
                print(f"  skip (unreadable): {path}")
                continue
            # Smaller resize and fewer iters for the batch path - GrabCut at
            # 800px is way too slow. Mask quality at 300px is fine for the
            # downstream HSV histogram masking.
            img = preprocess(img, max_side=300)
            mask, _ = grabcut_building(img, iters=1)
            hogs.append(hog_descriptor(img))
            hists.append(hsv_histogram(img, mask))
            _, des = orb_features(img)
            orbs.append(des if des is not None else np.zeros((0, 32), dtype=np.uint8))
            labels.append(cls_to_idx[cls])
            paths.append(path)
            if i % 25 == 0:
                print(f"  {cls}: {i}/{len(files)}", flush=True)
    return {
        "hog": np.asarray(hogs, dtype=np.float32),
        "hsv": np.asarray(hists, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "classes": classes,
        "paths": paths,
        "orb": orbs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--out", default="data/features")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val", "test"):
        sd = os.path.join(a.splits, split)
        if not os.path.isdir(sd):
            print(f"skip missing: {sd}")
            continue
        print(f"== {split} ==")
        d = process_split(sd)
        np.savez(
            os.path.join(a.out, f"{split}.npz"),
            hog=d["hog"], hsv=d["hsv"], labels=d["labels"],
            classes=np.array(d["classes"]), paths=np.array(d["paths"]),
        )
        with open(os.path.join(a.out, f"{split}_orb.pkl"), "wb") as fh:
            pickle.dump(d["orb"], fh)
        print(f"  saved {split}: hog={d['hog'].shape} hsv={d['hsv'].shape} n={len(d['labels'])}")


if __name__ == "__main__":
    main()
