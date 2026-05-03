"""Feature extraction: Canny+Hough, ORB, HOG, color histogram."""
import cv2
import numpy as np
from skimage.feature import hog


def canny_edges(img_bgr, low=60, high=180):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def hough_lines(edges, min_len_frac=0.10):
    """Probabilistic Hough on Canny edges. min line length scales with image size."""
    h, w = edges.shape
    min_len = int(min(h, w) * min_len_frac)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=10,
    )
    return lines if lines is not None else np.empty((0, 1, 4), dtype=np.int32)


def orb_features(img_bgr, n=500):
    """ORB: rotation+scale invariant keypoints. 256-bit binary descriptors."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n)
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des


def hog_descriptor(img_bgr, size=(128, 128)):
    """Global HOG vector. Resize to fixed size first so descriptor length is constant."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)


def hsv_histogram(img_bgr, mask=None, bins=(16, 16, 16)):
    """3D HSV histogram, normalized. Mask restricts to building region."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_all(img_bgr, mask=None):
    """Run every extractor and return a dict. `mask` is the building mask from segment.py."""
    edges = canny_edges(img_bgr)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    lines = hough_lines(edges)
    kps, des = orb_features(img_bgr)
    return {
        "edges": edges,
        "lines": lines,
        "orb_keypoints": kps,
        "orb_descriptors": des,
        "hog": hog_descriptor(img_bgr),
        "hsv_hist": hsv_histogram(img_bgr, mask),
    }
