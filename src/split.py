"""Stratified 60/20/20 split into data/splits/{train,val,test}/<class>/."""
import argparse
import os
import random
import shutil


def split_class(src_dir, out_root, cls, ratios=(0.6, 0.2, 0.2), seed=42):
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    parts = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for split, items in parts.items():
        dst_dir = os.path.join(out_root, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for f in items:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print(f"  {cls}/{split}: {len(items)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    classes = sorted(d for d in os.listdir(a.raw) if os.path.isdir(os.path.join(a.raw, d)))
    print(f"classes: {classes}")
    for cls in classes:
        split_class(os.path.join(a.raw, cls), a.out, cls, seed=a.seed)
    print("done.")


if __name__ == "__main__":
    main()
