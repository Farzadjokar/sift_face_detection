# SIFT Face Detection

A face detector built from classic computer-vision parts: **dense SIFT** features, a **bag of visual words**, and a **linear SVM**, with a simple **Tkinter** desktop app. Open an image and the app draws a bounding box around every face it finds, with no settings required.

## Features

- **Dense SIFT + Bag of Visual Words:** RootSIFT descriptors on a regular grid, quantised with a vocabulary learned by k-means
- **Linear SVM** over spatial word histograms (whole window + 3x3 grid) so the face layout is captured
- **Fast sliding window:** SIFT and word assignment are computed once per pyramid level, and window histograms come from integral images
- **Image pyramid**, so faces of any size are found
- **Automatic negatives:** non-face photos are streamed from COCO on Hugging Face (only images that contain no person), so you do not have to collect any
- **Hard-negative mining** and an automatically calibrated detection threshold
- **Tkinter app** that loads the model automatically and detects as soon as you open an image

## How it works

**Training (`train_classifier.py`)**

1. Loads your face images from the `face/` folder as positives (original, mirrored and slightly jittered copies).
2. Downloads person-free images from COCO (`detection-datasets/coco`, streamed) as negatives and saves them to `non_face_images/`. Misaligned windows cut from the face images (shifted, too small, or only part of a face) are added as extra negatives.
3. Learns a visual vocabulary (default 128 words) by clustering SIFT descriptors.
4. Describes every 64x64 window as Hellinger-normalised histograms of visual words and trains a `LinearSVC`.
5. Runs hard-negative mining: it scans person-free images, collects the windows the model wrongly accepts, and retrains.
6. Calibrates the detection threshold on held-out person-free images and saves everything to `face_sift_svm.npz`.

**Detection (`face_app.py` and `sift_core.py`)**

1. Converts the image to grayscale and builds an image pyramid.
2. For each level, computes dense SIFT descriptors and assigns each one to its nearest visual word.
3. Scores every 64x64 window with the SVM, using integral images of the word counts for speed.
4. Groups overlapping hits into faces (a face needs several overlapping hits, which removes most isolated false positives) and draws a green box with the score.

## Project structure

```
.
├── sift_core.py          # SIFT / visual-word window model, pyramid scoring, grouping
├── train_classifier.py   # training script
├── face_app.py           # Tkinter detection app
├── requirements.txt
├── face/                 # your face images (not in the repo)
├── non_face_images/      # downloaded automatically on the first training run
└── face_sift_svm.npz     # trained model, created by training
```

## Installation

```bash
git clone https://github.com/Farzadjokar/sift_face_detection.git
cd sift_face_detection
pip install -r requirements.txt
```

SIFT needs OpenCV 4.4 or newer. Tkinter ships with Python on Windows and macOS. On Debian/Ubuntu: `sudo apt install python3-tk`.

## Dataset setup

Put your face images in a folder named `face` next to the scripts. Subfolders are fine, and any common image format works:

```
face/
    img_0001.jpg
    img_0002.jpg
    ...
```

The images should be cropped faces. If your images are larger photos with a small face in the middle, train with `--face_center 0.5` (or whatever fraction of the image the face covers).

No negative dataset is needed. On the first run the script downloads about 1,500 non-face images automatically and reuses them afterwards. To use your own person-free photos instead, pass `--neg_dir /path/to/photos`.

## Train

```bash
python train_classifier.py
```

This creates `face_sift_svm.npz` in the current folder. Training prints a confusion matrix and precision / recall for the validation and test splits, and reports how many validation faces clear the calibrated threshold.

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--face_dir` | `face` | folder with face images |
| `--face_center` | 1.0 | use only the central fraction of each face image |
| `--max_faces` | 6000 | maximum number of face images used |
| `--neg_dir` | none | your own folder of person-free images (skips the download) |
| `--neg_images` | 1500 | number of COCO images to download |
| `--neg_per_image` | 30 | random windows sampled per negative image |
| `--partial_ratio` | 0.5 | misaligned-face negatives, as a fraction of the faces |
| `--vocab` | 128 | visual vocabulary size |
| `--mine_rounds` | 2 | hard-negative mining rounds |
| `--cal_quantile` | 0.9 | threshold calibration; lower it to miss fewer faces |
| `--C` | 0.5 | SVM regularisation |

## Run the app

```bash
python face_app.py
```

Click **Open Image**. The model is loaded automatically from the script's folder and detection starts immediately. **Save Result** stores the image with the boxes drawn on it. If `face_sift_svm.npz` is already in the folder, you can skip training.

### Tuning

Constants at the top of `face_app.py`:

| Constant | Effect |
| --- | --- |
| `THR_OFFSET` | raise it for fewer false faces, lower it (even negative) for fewer missed faces |
| `MIN_NEIGHBORS` | overlapping hits required to accept a face |
| `PYRAMID_STEP` | smaller = more pyramid levels (slower, finer scale coverage) |
| `DETECT_MAX_SIDE` | larger = finds smaller faces in big images (slower) |

## Results

Detection quality depends on your face images. Training prints window-level precision and recall for the validation and test splits. Judge the detector on real photos too, because window-level scores do not capture false boxes on full images.

## Limitations

- SIFT bag-of-visual-words is a classic method and is much weaker than Haar cascades, HOG detectors or modern CNN detectors. Expect some missed faces and some false boxes.
- The detector is trained on upright, roughly frontal faces. Profile faces, strong rotation and heavy occlusion are likely to be missed.
- The 64x64 window means very small faces are found only when the image is enlarged, which the app does for small images.
- The negatives are everyday photos, so unusual face-like patterns (posters, toys, textures) may still trigger false detections.

## References

- D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 (SIFT).
- G. Csurka et al., *Visual Categorization with Bags of Keypoints*, ECCV Workshop 2004 (bag of visual words).
- R. Arandjelovic and A. Zisserman, *Three Things Everyone Should Know to Improve Object Retrieval*, CVPR 2012 (RootSIFT).
- T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.