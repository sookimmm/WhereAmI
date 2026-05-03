# WhereAmI

CSE 40535 Computer Vision — Spring 2026 — Project 03

This is the first coding update for my semester project, "Where am I located?".
The goal is to take a phone photo of a Notre Dame building and figure out
which building it is. For this update I focused on getting the
preprocessing, segmentation, and feature extraction working. Classification
will come in the next project.

## Buildings I'm using

I picked three buildings on campus that are close to each other and have
distinct architecture:

- **Mendoza** (College of Business) — modern light stone with a big dark glass
  curtain wall.
- **Morris_Inn** — older brick and stone, arched entrance.
- **Welsh_Fam** — light beige brick with a glass entry tower.

In Project 02 I had said I would use Mendoza, Fitzpatrick, and Hesburgh, but
when I went out to actually shoot the photos I switched to these three
because the lighting was better that day and they were close enough to all
shoot in one trip.

## How to run on the sample images

I included one sample photo per building in `sample_data/` so the code can be
tested without downloading the full dataset.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/demo.py --image sample_data/Mendoza_sample.jpg    --out outputs/demo_mendoza.png
python src/demo.py --image sample_data/Morris_Inn_sample.jpg --out outputs/demo_morris.png
python src/demo.py --image sample_data/Welsh_Fam_sample.jpg  --out outputs/demo_welsh.png
```

This runs the whole pipeline (preprocess → segment → features) and saves a
2x3 grid showing each step.

To run on the full dataset, put the raw images in `data/raw/<class>/` and run:

```bash
python src/split.py             # 60/20/20 split into data/splits/
python src/extract_features.py  # writes feature vectors to data/features/
```

## File layout

```
src/
  preprocess.py        - resize, denoise, CLAHE
  segment.py           - GrabCut segmentation
  features.py          - Canny+Hough, ORB, HOG, HSV histogram
  split.py             - train/val/test split
  extract_features.py  - runs feature extraction on the whole dataset
  demo.py              - runs full pipeline on one image
sample_data/           - 3 sample photos (one per class)
outputs/               - demo PNGs
```

## Dataset

I took all the photos myself with my iPhone (4032x3024 JPGs). Following the
plan from Project 02, for each building I shot photos at different angles
(0°, ±30°, ±60°, ±90°), distances (near, medium, far), and tilts. The split
is 60% train / 20% val / 20% test, with a fixed random seed so it's
reproducible.

| Class | # images |
|---|---|
| Mendoza | 264 |
| Morris_Inn | 125 |
| Welsh_Fam | 264 |

The Morris_Inn count is lower because one of my zip files didn't upload
properly — I'm going to retake those before the next project.

---

# Report

## Methods I applied

**Preprocessing** (`src/preprocess.py`):
- Resize so the longest side is 800 pixels (with `cv2.INTER_AREA`).
- Bilateral filter for noise removal.
- CLAHE on the L channel of LAB color space for contrast normalization.

**Segmentation** (`src/segment.py`):
- GrabCut, initialized with a rectangle that's inset 8% from the image
  border.
- Morphological open + close to clean up the mask.

**Feature extraction** (`src/features.py`):
- Canny edge detection (thresholds 60 / 180).
- Probabilistic Hough line transform on the masked edges.
- ORB keypoints and descriptors (up to 500 per image).
- HOG descriptor on a 128x128 grayscale version of the image.
- 3D HSV color histogram (16x16x16 bins), only inside the building mask.

## Why I chose these methods

In Project 02 I wrote that the features should focus on the structure and
geometry of the buildings (edges, shapes, patterns) and not just color, and
that they need to be robust to changes in lighting, scale, and camera
angle. Every choice below comes from those two requirements.

**Bilateral filter + CLAHE.** I knew lighting was going to be a problem
because I shot on a sunny winter day with very harsh shadows. CLAHE on the L
channel evens out the local contrast without messing up the colors, which I
wanted to keep because the three buildings actually have different brick
and stone tones. I picked bilateral filter over a Gaussian blur because the
next step is Canny edge detection, and a Gaussian would smooth out the
window-frame edges I'm trying to detect. Bilateral keeps edges sharp while
still removing noise.

**GrabCut for segmentation.** My photos always have stuff I don't want in
them — sky, grass, sidewalk, sometimes people. I needed a way to focus on
the building. GrabCut is good for this because it doesn't need any training
data, and since I always tried to put the building in the middle of the
frame, an inset rectangle works well as the foreground prior. I considered
using simple thresholding to remove the sky, but the sky color changes a
lot depending on the weather, so it wasn't reliable. I also thought about
using a deep-learning segmentation model, but that felt like overkill for
3 classes and would have changed this from a CV class project to a deep
learning project.

**Canny + Hough.** This one is the example from the assignment prompt, but
it really does fit my project. All three buildings have a lot of straight
lines — window grids, doorways, mullions — and Hough lines pick those up
well. I run Canny first to get the edges, then mask the edges with the
GrabCut mask so I'm only finding lines on the building itself, and then run
Probabilistic Hough on what's left. The `minLineLength` parameter scales
with image size so the same code works at different resolutions.

**ORB.** Project 02 specifically said the features should be invariant to
rotation, scale, and viewpoint, which is exactly what ORB was designed for.
ORB descriptors are also binary (256 bits), so they're small and fast to
match later when I get to classification. I picked ORB instead of SIFT or
SURF because ORB is free to use and ships with OpenCV.

**HOG.** ORB is local — it only looks at small patches around the
keypoints. HOG looks at the whole image at once and captures the overall
gradient pattern. I figured having both a local and a global descriptor
would help the classifier in Project 04. I resize the image to 128x128
before computing HOG so the vector is the same length for every image,
which makes it easy to feed into a classifier.

**HSV histogram.** I know I said the features shouldn't be just color, but
I think color is still useful as one feature among several. The three
buildings really do have different masonry tones, and as long as I'm not
relying only on color it should be fine. To make sure the histogram isn't
picking up the sky or grass, I compute it inside the building mask only.

## Illustrations

I wrote a demo script (`src/demo.py`) that runs the whole pipeline on one
image and saves a 2x3 grid showing each step. I ran it on one sample from
each class and saved the results in `outputs/`:

- `outputs/demo_mendoza.png` — GrabCut isolates the central facade well,
  Hough finds 38 lines that trace the window grid and the entry portal,
  and ORB clusters its keypoints on the high-contrast glass mullions.
- `outputs/demo_welsh.png` — Hough finds about 118 lines tracing the brick
  rows and window frames, ORB keypoints are spread over the brick texture.
- `outputs/demo_morris.png` — the pipeline still runs but you can see
  GrabCut struggles a bit because the building isn't centered in this
  particular shot and the patterned patio takes up the bottom of the
  frame. This is something I want to fix before Project 04.

## Things that aren't perfect yet

- GrabCut assumes the building is centered. When it's not (like in the
  Morris_Inn sample) the foreground patio leaks into the mask. I might add
  a sky-removal step using HSV thresholding to give GrabCut a better starting
  point.
- The number of Hough lines varies a lot between images (38 vs 118 vs 250+).
  For the classifier I'll probably summarize them with statistics — count,
  dominant orientation, mean length — instead of using the raw lines.
- 500 ORB keypoints per image is too many to use directly, so I'll convert
  them to a fixed-length bag-of-visual-words vector before classification.
- Morris_Inn only has 125 images right now because of a failed upload. I
  need to reshoot or recover those before Project 04.

## Individual contributions

This is a solo project. I did all the data collection, code, and writing.
