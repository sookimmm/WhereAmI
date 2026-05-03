# WhereAmI — Notre Dame Building Recognition

CSE 40535 Computer Vision · Spring 2026 · Project 03 (First Coding Update)

A computer-vision system that takes a smartphone photo of a Notre Dame campus
building and predicts which building it is. This update covers everything
*before* classification: data preprocessing, segmentation, and feature
extraction. Classifier training and evaluation will land in Project 04.

## Buildings (3 classes)

| Class | Visual signature |
|---|---|
| **Mendoza** (College of Business) | modern light-stone facade, large dark glass curtain wall, sharp rectangular grid |
| **Morris_Inn** | brick + stone Gothic, arched entrance, varied window shapes |
| **Welsh_Fam** | light beige brick, reflective angled glass entry tower, "1997" engraving |

> Note: the original Project 02 proposal listed Mendoza / Fitzpatrick / Hesburgh.
> The class set was revised to **Mendoza / Morris_Inn / Welsh_Fam** after
> reshooting on campus, because these three sit close to each other and offered
> better lighting/angle variety for a single shoot.

## Repository layout

```
WhereAmI/
├── src/
│   ├── preprocess.py        # resize, denoise, CLAHE
│   ├── segment.py           # GrabCut → building mask
│   ├── features.py          # Canny+Hough, ORB, HOG, HSV histogram
│   ├── split.py             # 60/20/20 stratified split
│   ├── extract_features.py  # batch feature extraction → .npz / .pkl
│   └── demo.py              # full-pipeline visualization for one image
├── sample_data/             # 1 image per class (committed for graders)
├── outputs/                 # demo PNGs (committed)
├── requirements.txt
└── README.md                # ← this file
```

## How to run on a sample

The repo ships three sample images in `sample_data/` so a grader can run the
pipeline without any of the raw data.

```bash
# 1. set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the full pipeline on one sample image
python src/demo.py --image sample_data/Mendoza_sample.jpg     --out outputs/demo_mendoza.png
python src/demo.py --image sample_data/Morris_Inn_sample.jpg  --out outputs/demo_morris.png
python src/demo.py --image sample_data/Welsh_Fam_sample.jpg   --out outputs/demo_welsh.png
```

Each command produces a 2×3 grid showing original → preprocessed →
segmented → Canny edges → Hough lines → ORB keypoints.

To run on the full dataset (after you place raw class folders under
`data/raw/<class>/`):

```bash
python src/split.py                # 60/20/20 stratified split into data/splits/
python src/extract_features.py     # writes data/features/{train,val,test}.npz + *_orb.pkl
```

## Dataset

Photos taken personally on campus with an iPhone (4032×3024 RGB JPG).

| Class | # images |
|---|---|
| Mendoza | 264 |
| Morris_Inn | 125 (one of two zips lost in upload — will be re-shot) |
| Welsh_Fam | 264 |

Following the Project 02 protocol: each shot varies orientation (neutral / tilt
up / tilt down / ground-level), angle (0°, ±30°, ±60°, ±90° off perpendicular),
and distance (near / medium / far). Stratified random split: **60% train /
20% val / 20% test**, fixed seed=42 so results are reproducible.

---

# Report

## 1. Methods applied

### Preprocessing (`src/preprocess.py`)
- **Resize to longest-side = 800 px** with `cv2.INTER_AREA`
- **Bilateral filter** (`d=7`, σ_color=σ_space=50)
- **CLAHE on the L channel of LAB** (clip=2.0, 8×8 tiles)

### Segmentation (`src/segment.py`)
- **GrabCut** initialized with an inset rectangle (8% border) as
  probable-foreground prior, 3 iterations
- Morphological **open + close** (5×5 ellipse) to clean specks and holes

### Feature extraction (`src/features.py`)
- **Canny edge detection** (thresholds 60 / 180) on the grayscale image,
  masked to the building region
- **Probabilistic Hough line transform** (`HoughLinesP`) on the masked
  edge map; `minLineLength` scales with image size (10% of the short side)
- **ORB keypoints + 256-bit binary descriptors** (up to 500 per image)
- **Global HOG descriptor** on a 128×128 grayscale resize
  (9 orientations, 16-px cells, 2×2 block, L2-Hys norm) — 1764-d vector
- **3-D HSV color histogram**, 16×16×16 bins, normalized — 4096-d vector,
  computed inside the building mask

## 2. Justification

The Project 02 proposal called out two requirements that drove every choice
here: features should be **structural and geometric** rather than colour-only,
and they should be **robust to lighting / scale / orientation changes** while
**sensitive to architectural differences**. We chose four complementary
descriptors so that the eventual classifier sees the building from multiple
angles (literally and figuratively).

**Why bilateral + CLAHE for preprocessing?** Project 02 anticipated wide
lighting variation — clear sun vs. overcast, harsh shadows from low winter
sun. CLAHE on L-channel normalizes local contrast without distorting hue,
which matters because the three buildings *are* distinguishable partly by
masonry tone (warm Morris brick vs. cool Mendoza limestone). Bilateral
smoothing was preferred over Gaussian because the downstream Canny step
needs preserved edges; a Gaussian blur would soften the very window-frame
lines that drive recognition.

**Why GrabCut for segmentation?** The proposal noted that backgrounds
contain "people, trees, shadows and other buildings." On these images the
target building dominates the central frame (this was enforced during
shooting), so an inset rectangular prior gives GrabCut enough information
to converge on a clean foreground without manual annotation. Cheaper
alternatives were considered and rejected: thresholding fails because sky
luminance varies with weather; semantic segmentation networks were
overkill for three classes and would shift the project from a CV-feature
exercise to a deep-learning exercise. GrabCut is unsupervised, classical,
and matches the spirit of the assignment.

**Why Canny + Hough?** This is the example explicitly mentioned in the
Project 03 prompt, but it also fits the data: every one of these buildings
has strong rectilinear structure (window grids, mullions, doorways,
cornices) that a line-based descriptor captures well. Masking edges with
the building mask before Hough cuts down on spurious lines from
foreground clutter (sidewalk seams, branches) — visible in the
`outputs/demo_morris.png` panel where the patio paving still produces some
noise; that is an honest weakness when the building is small in frame.

**Why ORB?** Project 02 called for **rotation, scale, and viewpoint
invariance** — exactly what ORB was designed for. ORB descriptors are
binary (256-bit), which keeps storage small (500×32 bytes per image) and
makes Hamming-distance matching very fast in the classifier stage.
Compared to SIFT/SURF, ORB is patent-free and ships with OpenCV; compared
to BRIEF/FAST alone, ORB adds rotation invariance via oriented keypoints.

**Why HOG?** ORB is local; HOG is global. A holistic descriptor of the
overall gradient structure complements the keypoint-bag-of-features view
by forcing the classifier to look at the *layout* of the building, not
just isolated patches. The fixed 128×128 resize means HOG vectors are the
same length for every image, so they drop straight into a linear
classifier.

**Why HSV histogram if "not colour alone"?** The proposal said colour
shouldn't be the *primary* signal, not that it should be discarded. The
three buildings really do have distinct masonry tones, and including HSV
as one feature among four lets the classifier weight it appropriately.
Computing the histogram inside the building mask is what makes this safe:
sky and grass — which depend on weather and season — are excluded.

## 3. Illustrations

Generated by running `python src/demo.py --image <path>`; full PNGs live in
`outputs/`. Each grid shows: original → preprocessed → segmented → masked
Canny edges → Hough lines (count) → ORB keypoints (count).

| Building | Pipeline output |
|---|---|
| Mendoza | `outputs/demo_mendoza.png` — GrabCut isolates the central facade; Hough catches 38 strong lines tracing the window grid and entry portal; ORB clusters 500 keypoints on the high-contrast glass-mullion intersections. |
| Welsh_Fam | `outputs/demo_welsh.png` — Hough returns ~118 lines tracing brick-course rows and window frames; ORB keypoints populate the brick texture densely. |
| Morris_Inn | `outputs/demo_morris.png` — pipeline still completes, but the patterned patio dominates the lower frame, illustrating that GrabCut's rectangular prior weakens when the building is not centered. A future iteration may add a sky-segmentation pre-pass to shrink the search rect. |

## 4. Known limitations / next steps

- GrabCut's inset-rectangle prior assumes the building is roughly centered.
  Off-center shots (Morris_Inn) leak foreground patio into the mask. Adding
  an HSV-based sky removal as a hard background prior would help.
- Hough line count varies wildly across viewpoints (38–250+); the Project
  04 classifier should consume Hough output as **summary statistics**
  (count, dominant orientation histogram, mean length) rather than raw
  line lists.
- 500 ORB keypoints per image will be aggregated into a fixed-length
  bag-of-visual-words vector before classification.
- Morris_Inn is currently undersized (125 vs 264 in the other classes) due
  to a lost upload; this will be rebalanced before Project 04.

## 5. Individual contributions

Solo project — Soo Kim — all data collection, code, and report.
