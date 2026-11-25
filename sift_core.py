"""
Shared code for the SIFT face detector (used by train_classifier.py and face_app.py).

Method (Bag of Visual Words on dense SIFT):
  1. Dense SIFT: a RootSIFT descriptor is computed on a regular grid (one every STEP px).
  2. Every descriptor is assigned to the nearest word of a learned visual vocabulary.
  3. A 64x64 window is described by histograms of visual words over spatial regions
     (whole window + 3x3 grid), Hellinger-normalised (square root).
  4. A linear SVM scores the window.
For detection the SIFT grid and word assignment are computed ONCE per pyramid level, and
window histograms are obtained from integral images, so scoring every window is fast.
"""
from pathlib import Path

import cv2
import numpy as np

WIN = 64                 # detection window (square, pixels)
STEP = 8                 # grid spacing = cell size (pixels)
CELLS = WIN // STEP      # 8 cells per window side
SIFT_SIZE = 4            # SIFT keypoint size -> small (~24 px) descriptor patches
PAD = 24                 # reflect padding around an image so border keypoints are valid
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm", ".tif", ".tiff")

# Feature regions in window-cell coordinates (r0, r1, c0, c1).
# The outer ring of cells is NOT used, so descriptors stay (almost) inside the window and
# training crops and real image windows look the same.
_I0, _I1 = 1, CELLS - 1
REGIONS = [(_I0, _I1, _I0, _I1)]
for _a in range(3):
    for _b in range(3):
        REGIONS.append((_I0 + 2 * _a, _I0 + 2 * _a + 2, _I0 + 2 * _b, _I0 + 2 * _b + 2))
NREG = len(REGIONS)
AREAS = [float((r1 - r0) * (c1 - c0)) for r0, r1, c0, c1 in REGIONS]

_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def load_gray(path):
    data = np.fromfile(str(path), dtype=np.uint8)   # works with non-ASCII paths
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def resize_win(img):
    interp = cv2.INTER_AREA if min(img.shape[:2]) >= WIN else cv2.INTER_LINEAR
    return cv2.resize(img, (WIN, WIN), interpolation=interp)


# --------------------------------------------------------------------------- #
# Dense SIFT + visual words
# --------------------------------------------------------------------------- #
def grid_desc(gray, gh, gw):
    """RootSIFT descriptors for a gh x gw grid of STEP-px cells -> (gh*gw, 128) float32."""
    gray = np.ascontiguousarray(gray)
    padded = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    half = STEP / 2.0
    kps = [cv2.KeyPoint(float(PAD + half + STEP * j), float(PAD + half + STEP * i),
                        float(SIFT_SIZE), 0.0)
           for i in range(gh) for j in range(gw)]
    kps, desc = _sift().compute(padded, kps)
    if desc is None or len(kps) != gh * gw:
        raise RuntimeError("SIFT returned an unexpected number of descriptors")
    desc = desc.astype(np.float32)
    desc /= desc.sum(axis=1, keepdims=True) + 1e-7   # RootSIFT: L1 normalise ...
    np.sqrt(desc, out=desc)                           # ... then square root
    return desc


def assign(desc, vocab, vsq):
    """Nearest visual word for each descriptor."""
    d = vsq[None, :] - 2.0 * (desc @ vocab.T)
    return d.argmin(axis=1).astype(np.int32)


def window_feature(words, k):
    """Feature vector of one window from its (CELLS x CELLS) word indices."""
    parts = []
    for (r0, r1, c0, c1), area in zip(REGIONS, AREAS):
        cnt = np.bincount(words[r0:r1, c0:c1].ravel(), minlength=k).astype(np.float32)
        parts.append(np.sqrt(cnt / area))
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FaceModel:
    def __init__(self, vocab, w, b, thr=0.0):
        self.vocab = np.ascontiguousarray(vocab, dtype=np.float32)
        self.vsq = (self.vocab ** 2).sum(axis=1)
        self.k = len(self.vocab)
        self.w = np.asarray(w, dtype=np.float32).reshape(NREG, self.k)
        self.b = float(b)
        self.thr = float(thr)

    def sample_words(self, gray64):
        """Visual words (CELLS x CELLS) of a 64x64 gray sample."""
        d = grid_desc(gray64, CELLS, CELLS)
        return assign(d, self.vocab, self.vsq).reshape(CELLS, CELLS)

    def save(self, path):
        np.savez(path, vocab=self.vocab, w=self.w.ravel(), b=np.float32(self.b),
                 thr=np.float32(self.thr), version=np.int32(1))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["vocab"], d["w"], float(d["b"]), float(d["thr"]))


# --------------------------------------------------------------------------- #
# Scoring all windows of one pyramid level
# --------------------------------------------------------------------------- #
def score_map(words, model):
    """SVM score of every window (stride = one cell) from the grid of word indices."""
    gh, gw = words.shape
    nwy, nwx = gh - CELLS + 1, gw - CELLS + 1
    if nwy <= 0 or nwx <= 0:
        return None
    k = model.k
    onehot = np.zeros((k, gh + 1, gw + 1), dtype=np.float32)
    ys, xs = np.indices((gh, gw))
    onehot[words.ravel(), ys.ravel() + 1, xs.ravel() + 1] = 1.0
    integral = onehot.cumsum(axis=1).cumsum(axis=2)

    score = np.full((nwy, nwx), model.b, dtype=np.float32)
    for r, ((r0, r1, c0, c1), area) in enumerate(zip(REGIONS, AREAS)):
        cnt = (integral[:, r1:r1 + nwy, c1:c1 + nwx] - integral[:, r0:r0 + nwy, c1:c1 + nwx]
               - integral[:, r1:r1 + nwy, c0:c0 + nwx] + integral[:, r0:r0 + nwy, c0:c0 + nwx])
        score += np.tensordot(model.w[r], np.sqrt(cnt / area), axes=(0, 0))
    return score


def iter_levels(gray, model, step=1.2, max_side=960, max_up=2.0):
    """Yield (scale, score_map) for each level of the image pyramid."""
    h, w = gray.shape
    s = min(max_up, max_side / max(h, w))
    while True:
        nw, nh = int(round(w * s)), int(round(h * s))
        if nw < WIN or nh < WIN:
            return
        if abs(s - 1.0) < 1e-6:
            level = gray
        else:
            interp = cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA
            level = cv2.resize(gray, (nw, nh), interpolation=interp)
        gh, gw = nh // STEP, nw // STEP
        words = assign(grid_desc(level, gh, gw), model.vocab, model.vsq).reshape(gh, gw)
        sc = score_map(words, model)
        if sc is not None:
            yield s, sc
        s /= step


def detect_raw(gray, model, thr, step=1.2, max_side=960, max_up=2.0, max_per_level=400):
    """All windows scoring above thr, as boxes [x, y, w, h] in image coordinates."""
    boxes, scores = [], []
    for s, sc in iter_levels(gray, model, step, max_side, max_up):
        ys, xs = np.nonzero(sc > thr)
        if len(ys) > max_per_level:
            top = np.argpartition(-sc[ys, xs], max_per_level)[:max_per_level]
            ys, xs = ys[top], xs[top]
        for y, x in zip(ys, xs):
            boxes.append([x * STEP / s, y * STEP / s, WIN / s, WIN / s])
            scores.append(float(sc[y, x]))
    return boxes, scores


def max_window_score(gray, model, **kw):
    """Highest window score anywhere in the image (used to calibrate the threshold)."""
    best = -1e9
    for _, sc in iter_levels(gray, model, **kw):
        best = max(best, float(sc.max()))
    return best


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _overlap(b, arr):
    """IoU and intersection area of box b=(x,y,w,h) with each row of arr."""
    x1 = np.maximum(b[0], arr[:, 0])
    y1 = np.maximum(b[1], arr[:, 1])
    x2 = np.minimum(b[0] + b[2], arr[:, 0] + arr[:, 2])
    y2 = np.minimum(b[1] + b[3], arr[:, 1] + arr[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = b[2] * b[3] + arr[:, 2] * arr[:, 3] - inter
    return inter / np.maximum(union, 1e-9), inter


def nms(boxes, scores, iou_thr=0.4, max_keep=20):
    """Plain greedy non-maximum suppression; returns kept indices."""
    B = np.array(boxes, dtype=np.float64)
    order = np.argsort(-np.array(scores))
    keep = []
    while order.size and len(keep) < max_keep:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        iou, _ = _overlap(B[i], B[rest])
        order = rest[iou < iou_thr]
    return keep


def group_detections(boxes, scores, thr, min_neighbors=2, group_iou=0.3, contain_thr=0.7):
    """
    Merge window hits into faces. A real face produces several overlapping hits
    (neighbouring positions / pyramid levels); isolated hits are treated as noise.
    Returns a list of (box[x,y,w,h], best_score, n_hits).
    """
    if not boxes:
        return []
    B = np.array(boxes, dtype=np.float64)
    S = np.array(scores, dtype=np.float64)
    used = np.zeros(len(B), dtype=bool)
    clusters = []
    for i in np.argsort(-S):
        if used[i]:
            continue
        iou, _ = _overlap(B[i], B)
        members = (~used) & (iou > group_iou)
        members[i] = True
        used |= members
        if members.sum() < min_neighbors:
            continue
        wts = S[members] - thr + 0.1
        box = (B[members] * wts[:, None]).sum(0) / wts.sum()
        clusters.append((box, float(S[members].max()), int(members.sum())))

    clusters.sort(key=lambda c: -c[1])
    kept = []
    for box, sc, n in clusters:
        area = box[2] * box[3]
        inside = False
        for kb, _, _ in kept:
            _, inter = _overlap(box, kb[None, :])
            if inter[0] / max(min(area, kb[2] * kb[3]), 1e-9) > contain_thr:
                inside = True
                break
        if not inside:
            kept.append((box, sc, n))
    return kept