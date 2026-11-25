"""
Train a SIFT (bag of visual words) + linear SVM face detector.

  Positives: your face images, in the folder  ./face   (searched recursively)
  Negatives: downloaded automatically (first run) from the COCO dataset on Hugging Face
             (`detection-datasets/coco`, streamed): only images that contain NO person.
             They are saved to ./non_face_images and reused on later runs.
             Or use your own person-free photos with --neg_dir.

Usage (from the project folder):
    python train_classifier.py

Output:
    face_sift_svm.npz   vocabulary + SVM weights + calibrated threshold
                        (face_app.py loads it automatically)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from sift_core import (CELLS, WIN, FaceModel, _I0, _I1, assign, detect_raw, grid_desc,
                       list_images, load_gray, max_window_score, nms, resize_win,
                       window_feature)


# --------------------------------------------------------------------------- #
# Negative images (automatic download)
# --------------------------------------------------------------------------- #
def download_negatives(cache_dir: Path, n_images: int, repo: str):
    """Stream COCO and keep the first n_images that contain no 'person'."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = list_images(cache_dir)
    if len(existing) >= max(50, int(0.5 * n_images)):
        print(f"Using {len(existing)} cached non-face images in {cache_dir}")
        return cache_dir

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Missing package: run `pip install datasets` "
                         "(or put person-free photos in a folder and use --neg_dir).")

    print(f"Streaming '{repo}' from Hugging Face and keeping images WITHOUT people ...")
    ds = load_dataset(repo, split="train", streaming=True)

    person_id = 0
    try:
        feats = ds.features["objects"]
        cat_feat = feats["category"] if "category" in feats else feats["categories"]
        names = cat_feat.feature.names
        person_id = names.index("person")
    except Exception:
        print("  (could not read class names; assuming 'person' has id 0)")

    saved = seen = 0
    for ex in ds:
        seen += 1
        objs = ex["objects"]
        cats = objs.get("category", objs.get("categories", []))
        if person_id not in cats:
            img = ex["image"].convert("RGB")
            w, h = img.size
            m = max(w, h)
            if m > 800:
                img = img.resize((int(w * 800 / m), int(h * 800 / m)))
            img.save(cache_dir / f"coco_{saved:05d}.jpg", quality=92)
            saved += 1
            if saved % 100 == 0:
                print(f"  saved {saved}/{n_images} (scanned {seen})")
            if saved >= n_images:
                break
        if seen > 30000:
            break

    print(f"Saved {saved} person-free images to {cache_dir}")
    if saved < 20:
        raise SystemExit("Could not collect enough negative images; use --neg_dir instead.")
    return cache_dir


# --------------------------------------------------------------------------- #
# Sample generators (all return 64x64 gray uint8 windows)
# --------------------------------------------------------------------------- #
def _square_crop(gray, side, dy=0, dx=0):
    h, w = gray.shape
    y0 = int(np.clip(h // 2 - side // 2 + dy, 0, max(0, h - side)))
    x0 = int(np.clip(w // 2 - side // 2 + dx, 0, max(0, w - side)))
    return gray[y0:y0 + side, x0:x0 + side]


def face_samples(path, seed, center):
    """Face crop, its mirror image, and one randomly jittered copy."""
    gray = load_gray(path)
    if gray is None:
        return []
    side = int(min(gray.shape) * center)
    if side < 16:
        return []
    rng = random.Random(seed)
    base = resize_win(_square_crop(gray, side))
    out = [base, cv2.flip(base, 1)]
    sd = max(16, int(side * rng.uniform(0.85, 1.0)))
    j = max(1, int(side * 0.06))
    jit = resize_win(_square_crop(gray, sd, rng.randint(-j, j), rng.randint(-j, j)))
    if rng.random() < 0.5:
        jit = cv2.flip(jit, 1)
    out.append(jit)
    return out


def partial_samples(path, seed, center):
    """Misaligned windows from a face image (shifted, too small, or only part of a face)."""
    gray = load_gray(path)
    if gray is None:
        return []
    side = int(min(gray.shape) * center)
    if side < 16:
        return []
    rng = random.Random(seed)
    face = _square_crop(gray, side)
    mode = rng.choice(["shift", "small", "zoom"])
    if mode == "shift":
        m = max(2, int(side * rng.uniform(0.3, 0.6)))
        big = cv2.copyMakeBorder(face, m, m, m, m, cv2.BORDER_REPLICATE)
        oy, ox = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)])
        crop = big[m + oy * m:m + oy * m + side, m + ox * m:m + ox * m + side]
    elif mode == "small":
        f = rng.uniform(0.35, 0.65)
        pad = int(side * (1.0 / f - 1.0) / 2)
        crop = cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    else:
        sub = max(12, int(side * rng.uniform(0.4, 0.6)))
        y = rng.randint(0, side - sub)
        x = rng.randint(0, side - sub)
        crop = face[y:y + sub, x:x + sub]
    return [resize_win(crop)]


def neg_samples(path, n, seed):
    """n random square windows (log-uniform size) from a person-free image."""
    gray = load_gray(path)
    if gray is None:
        return []
    h, w = gray.shape
    if min(h, w) < WIN:
        s = WIN / min(h, w)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        h, w = gray.shape
    rng = random.Random(seed)
    smax = min(h, w) / WIN
    out = []
    for _ in range(n):
        s = float(np.exp(rng.uniform(0.0, np.log(smax)))) if smax > 1.0 else 1.0
        side = min(int(WIN * s), h, w)
        y = rng.randint(0, h - side)
        x = rng.randint(0, w - side)
        out.append(resize_win(gray[y:y + side, x:x + side]))
    return out


# --------------------------------------------------------------------------- #
# Parallel workers
# --------------------------------------------------------------------------- #
def extract(kind, path, arg, seed, center, vocab, take):
    """
    vocab is None  -> return a few random inner-cell descriptors (to learn the vocabulary)
    otherwise      -> return the visual-word grid of every sample
    """
    if kind == "face":
        samples = face_samples(path, seed, center)[:arg]
    elif kind == "partial":
        samples = partial_samples(path, seed, center)
    else:
        samples = neg_samples(path, arg, seed)
    if not samples:
        return []

    if vocab is None:
        rng = np.random.RandomState(seed % (2 ** 31))
        chunks = []
        for s in samples:
            d = grid_desc(s, CELLS, CELLS).reshape(CELLS, CELLS, 128)[_I0:_I1, _I0:_I1].reshape(-1, 128)
            idx = rng.choice(len(d), min(take, len(d)), replace=False)
            chunks.append(d[idx])
        return [np.concatenate(chunks)]

    vsq = (vocab ** 2).sum(axis=1)
    return [assign(grid_desc(s, CELLS, CELLS), vocab, vsq).reshape(CELLS, CELLS).astype(np.uint8)
            for s in samples]


def mine_image(path, model, thr, max_keep):
    """False-positive windows of the current model on a person-free image."""
    gray = load_gray(path)
    if gray is None:
        return []
    boxes, scores = detect_raw(gray, model, thr)
    if not boxes:
        return []
    out = []
    for i in nms(boxes, scores, 0.4, max_keep):
        x, y, bw, bh = [int(round(v)) for v in boxes[i]]
        x, y = max(0, x), max(0, y)
        crop = gray[y:y + bh, x:x + bw]
        if crop.shape[0] >= 8 and crop.shape[1] >= 8:
            out.append(model.sample_words(resize_win(crop)).astype(np.uint8))
    return out


def image_max_score(path, model):
    gray = load_gray(path)
    return [max_window_score(gray, model)] if gray is not None else []


def run(fn, items, n_jobs, desc):
    print(f"  {desc} ({len(items)} items)")
    res = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(fn)(*it) for it in items)
    return [x for r in res for x in r]


def feats(words_list, k):
    return np.array([window_feature(w.astype(np.int32), k) for w in words_list], dtype=np.float32)


def report(name, clf, Xp, Xn):
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    pred = clf.predict(X)
    print(f"\n===== {name} =====")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["non-face", "face"], digits=4))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_dir", default=None, help="folder with face images (default: ./face)")
    ap.add_argument("--face_center", type=float, default=1.0,
                    help="use only the central fraction of each face image (e.g. 0.5 if faces "
                         "are small in the middle of larger photos)")
    ap.add_argument("--max_faces", type=int, default=6000)
    ap.add_argument("--neg_dir", default=None, help="your own folder of person-free images")
    ap.add_argument("--neg_cache", default="non_face_images")
    ap.add_argument("--neg_images", type=int, default=1500, help="how many images to download")
    ap.add_argument("--hf_dataset", default="detection-datasets/coco")
    ap.add_argument("--neg_per_image", type=int, default=30)
    ap.add_argument("--partial_ratio", type=float, default=0.5,
                    help="misaligned-face negatives as a fraction of #faces")
    ap.add_argument("--vocab", type=int, default=128, help="visual vocabulary size")
    ap.add_argument("--mine_rounds", type=int, default=2)
    ap.add_argument("--mine_per_image", type=int, default=15)
    ap.add_argument("--mine_thr", type=float, default=0.0)
    ap.add_argument("--cal_quantile", type=float, default=0.9,
                    help="threshold = this quantile of the best window score on person-free images")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--out", default="face_sift_svm.npz")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- faces ----
    if args.face_dir:
        face_dir = Path(args.face_dir)
    else:
        face_dir = next((Path(n) for n in ("face", "Face", "faces", "Faces") if Path(n).is_dir()), None)
    if face_dir is None or not face_dir.is_dir():
        raise SystemExit("Face folder not found. Put your images in ./face or pass --face_dir.")
    faces = list_images(face_dir)
    if len(faces) < 20:
        raise SystemExit(f"Only {len(faces)} images found in {face_dir}.")
    random.shuffle(faces)
    faces = faces[:args.max_faces]
    a, b = int(0.8 * len(faces)), int(0.9 * len(faces))
    f_tr, f_va, f_te = faces[:a], faces[a:b], faces[b:]
    print(f"faces: train {len(f_tr)}, val {len(f_va)}, test {len(f_te)}   ({face_dir})")

    # ---- non-face images ----
    neg_folder = Path(args.neg_dir) if args.neg_dir else download_negatives(
        Path(args.neg_cache), args.neg_images, args.hf_dataset)
    negs = list_images(neg_folder)
    if len(negs) < 20:
        raise SystemExit(f"Only {len(negs)} images in {neg_folder}.")
    random.shuffle(negs)
    a, b = int(0.7 * len(negs)), int(0.85 * len(negs))
    n_tr, n_va, n_te = negs[:a], negs[a:b], negs[b:]
    print(f"non-face images: train {len(n_tr)}, val {len(n_va)}, test {len(n_te)}")
    c = args.face_center

    # ---- 1. visual vocabulary ----
    print("\n[1/4] Learning the SIFT vocabulary ...")
    items = [("face", p, 3, i, c, None, 4) for i, p in enumerate(f_tr[:1500])]
    items += [("neg", p, 8, 100_000 + i, c, None, 4) for i, p in enumerate(n_tr[:600])]
    descs = np.vstack(run(extract, items, args.n_jobs, "sampling descriptors"))
    k = min(args.vocab, len(descs) // 20)
    print(f"  clustering {len(descs)} descriptors into {k} words")
    km = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, max_iter=100,
                         random_state=args.seed).fit(descs)
    vocab = km.cluster_centers_.astype(np.float32)

    # ---- 2. features ----
    print("\n[2/4] Extracting window features ...")
    W = run(extract, [("face", p, 3, i, c, vocab, 0) for i, p in enumerate(f_tr)],
            args.n_jobs, "train faces (x3: original, mirror, jitter)")
    Xp = feats(W, k)
    W = run(extract, [("neg", p, args.neg_per_image, 200_000 + i, c, vocab, 0) for i, p in enumerate(n_tr)],
            args.n_jobs, "train non-face windows")
    Xn = feats(W, k)
    n_part = int(args.partial_ratio * len(f_tr))
    if n_part:
        W = run(extract, [("partial", p, 1, 300_000 + i, c, vocab, 0) for i, p in enumerate(f_tr[:n_part])],
                args.n_jobs, "misaligned-face negatives")
        Xn = np.vstack([Xn, feats(W, k)])

    Xp_va = feats(run(extract, [("face", p, 1, 400_000 + i, c, vocab, 0) for i, p in enumerate(f_va)],
                      args.n_jobs, "val faces"), k)
    Xn_va = feats(run(extract, [("neg", p, 15, 500_000 + i, c, vocab, 0) for i, p in enumerate(n_va)],
                      args.n_jobs, "val non-face windows"), k)
    Xp_te = feats(run(extract, [("face", p, 1, 600_000 + i, c, vocab, 0) for i, p in enumerate(f_te)],
                      args.n_jobs, "test faces"), k)
    Xn_te = feats(run(extract, [("neg", p, 15, 700_000 + i, c, vocab, 0) for i, p in enumerate(n_te)],
                      args.n_jobs, "test non-face windows"), k)
    print(f"train: {len(Xp)} face / {len(Xn)} non-face samples, feature dim {Xp.shape[1]}")

    def fit(Xp_, Xn_):
        X = np.vstack([Xp_, Xn_])
        y = np.r_[np.ones(len(Xp_)), np.zeros(len(Xn_))]
        return LinearSVC(C=args.C, dual=False, max_iter=5000, class_weight="balanced").fit(X, y)

    # ---- 3. SVM + hard-negative mining ----
    print("\n[3/4] Training the SVM ...")
    clf = fit(Xp, Xn)
    report("VAL (round 0)", clf, Xp_va, Xn_va)

    for r in range(1, args.mine_rounds + 1):
        print(f"\nHard-negative mining, round {r}/{args.mine_rounds} ...")
        model = FaceModel(vocab, clf.coef_.ravel(), clf.intercept_[0])
        hard = run(mine_image, [(p, model, args.mine_thr, args.mine_per_image) for p in n_tr],
                   args.n_jobs, "scanning person-free images")
        print(f"  mined {len(hard)} false-positive windows")
        if len(hard) < 20:
            print("  (almost none left, stopping)")
            break
        Xn = np.vstack([Xn, feats(hard, k)])
        clf = fit(Xp, Xn)
        report(f"VAL (after mining round {r})", clf, Xp_va, Xn_va)

    report("TEST", clf, Xp_te, Xn_te)

    # ---- 4. calibrate the detection threshold ----
    print("\n[4/4] Calibrating the detection threshold ...")
    model = FaceModel(vocab, clf.coef_.ravel(), clf.intercept_[0])
    cal = n_va[:80] if len(n_va) >= 10 else n_tr[:80]
    maxes = np.array(run(image_max_score, [(p, model) for p in cal], args.n_jobs,
                         "best window score on person-free images"))
    thr = max(0.0, float(np.quantile(maxes, args.cal_quantile)))
    face_scores = clf.decision_function(Xp_va)
    print(f"  threshold = {thr:.3f}  ({args.cal_quantile:.0%} of person-free images have no window above it)")
    print(f"  validation faces above threshold: {(face_scores > thr).mean():.1%}")
    model.thr = thr
    model.save(args.out)
    print(f"\nSaved {args.out}")
    print("Tip: if too many faces are missed, lower --cal_quantile (e.g. 0.7) and retrain,\n"
          "     or lower THR_OFFSET in face_app.py.")


if __name__ == "__main__":
    main()