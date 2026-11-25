"""
Tkinter face detector (dense SIFT + bag of visual words + linear SVM).

- The model (face_sift_svm.npz from train_classifier.py) is loaded automatically from the
  script's folder or the current folder.
- Click "Open Image": faces are detected immediately and drawn with bounding boxes.
- An image pyramid is used, so faces of any size are found.

Run:  python face_app.py        (sift_core.py must be in the same folder)
Requires: opencv-python (>= 4.4), numpy, pillow
"""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from sift_core import FaceModel, detect_raw, group_detections

MODEL_NAME = "face_sift_svm.npz"

# Fixed defaults (edit here only if you want to tune)
THR_OFFSET = 0.0         # added to the calibrated threshold: raise = fewer false faces
MIN_NEIGHBORS = 2        # overlapping hits needed to accept a face
PYRAMID_STEP = 1.2       # shrink factor between pyramid levels
DETECT_MAX_SIDE = 960    # pyramid starts with the longest side at (at most) this size
MAX_UPSCALE = 2.0        # small images are enlarged up to this factor


def find_model():
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        p = folder / MODEL_NAME
        if p.is_file():
            return p
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SIFT Face Detector")
        self.geometry("1000x700")

        self.model = None
        self.image = None
        self.result = None
        self._tk_img = None
        self.busy = False
        self._job = None

        self._build_ui()
        self.after(100, self._auto_load_model)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=3)
        self.status = ttk.Label(bar, text="Loading model...", font=("TkDefaultFont", 10, "bold"))
        self.status.pack(side=tk.RIGHT, padx=8)

        self.canvas = tk.Canvas(self, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show(
            self.result if self.result is not None else self.image))

    def _auto_load_model(self):
        path = find_model()
        if path is None:
            searched = f"{Path(__file__).resolve().parent}\nand\n{Path.cwd()}"
            messagebox.showwarning(
                "Model not found",
                f"Could not find {MODEL_NAME} in:\n{searched}\n\n"
                "Run train_classifier.py first, or select the model file now.")
            chosen = filedialog.askopenfilename(title="Select trained model",
                                                filetypes=[("Model", "*.npz")])
            if not chosen:
                self.status.config(text="No model loaded", foreground="red")
                return
            path = Path(chosen)
        try:
            self.model = FaceModel.load(path)
            self.status.config(text=f"Model: {path.name}", foreground="green")
        except Exception as e:
            self.status.config(text="Model failed to load", foreground="red")
            messagebox.showerror("Model error", str(e))

    # ----------------------------------------------------------- detection
    def detect(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr = self.model.thr + THR_OFFSET
        boxes, scores = detect_raw(gray, self.model, thr, PYRAMID_STEP,
                                   DETECT_MAX_SIDE, MAX_UPSCALE)
        found = group_detections(boxes, scores, thr, MIN_NEIGHBORS)

        out = img.copy()
        thick = max(2, int(round(max(h, w) / 400)))
        for box, sc, _ in found:
            x, y, bw, bh = [int(round(v)) for v in box]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), thick)
            cv2.putText(out, f"{sc:.2f}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * thick / 2 + 0.3, (0, 255, 0), max(1, thick // 2), cv2.LINE_AA)
        return out, len(found)

    def _run_detection(self):
        """Detection runs in a worker thread so the window stays responsive."""
        self.busy = True
        self._job = None
        img = self.image

        def work():
            try:
                self._job = ("ok",) + self.detect(img)
            except Exception as e:
                self._job = ("err", str(e))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, self._poll_detection)

    def _poll_detection(self):
        if self._job is None:
            self.after(50, self._poll_detection)
            return
        self.busy = False
        if self._job[0] == "ok":
            _, self.result, n = self._job
            self.status.config(text=f"Faces detected: {n}", foreground="green")
            self._show(self.result)
        else:
            self.status.config(text="Detection failed", foreground="red")
            messagebox.showerror("Detection error", self._job[1])

    # --------------------------------------------------------------- image
    def open_image(self):
        if self.busy:
            return
        if self.model is None:
            self._auto_load_model()
            if self.model is None:
                return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.image = img
        self.result = None
        self._show(img)
        self.status.config(text="Detecting...", foreground="black")
        self._run_detection()  # automatic

    def _show(self, img):
        if img is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        disp = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "Open an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            ok, buf = cv2.imencode("." + path.rsplit(".", 1)[-1], self.result)
            if ok:
                buf.tofile(path)


if __name__ == "__main__":
    App().mainloop()