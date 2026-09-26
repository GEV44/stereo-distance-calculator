"""
Batch labelling tool (OpenCV): click corresponding points in stereo pairs
from ``left/`` and ``right/`` and type the measured distance.

Output: ``training_data.json`` — labelled records readable by
``python -m stereo_gamma train --points training_data.json``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DISP_W, DISP_H = 800, 600
WINDOW = "Collect points | L then R, type distance in terminal | N/P pair  U undo  Q quit"


def build_pairs(left_dir: str, right_dir: str):
    """Pair images by name (``left`` ↔ ``right`` substitution), else by sorted order."""
    lf = sorted(f for f in os.listdir(left_dir) if f.lower().endswith(IMAGE_EXT))
    rf = sorted(f for f in os.listdir(right_dir) if f.lower().endswith(IMAGE_EXT))
    pairs, rset = [], set(rf)
    for f in lf:
        g = f.replace("left", "right").replace("LEFT", "RIGHT").replace("Left", "Right")
        if g in rset:
            pairs.append((os.path.join(left_dir, f), os.path.join(right_dir, g)))
    if not pairs:
        pairs = [(os.path.join(left_dir, a), os.path.join(right_dir, b)) for a, b in zip(lf, rf)]
    return pairs


class Collector:
    def __init__(self, pairs, out_path):
        self.pairs, self.out = pairs, out_path
        self.data = self._load()
        self.i = 0
        self.pending: dict = {}
        self.scale = 1.0
        self.left_w = 0
        self.canvas = None

    def _load(self):
        if os.path.exists(self.out):
            try:
                with open(self.out, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"[warn] {self.out} is empty or corrupt — starting fresh")
        return []

    def save(self):
        with open(self.out, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        print(f"[saved] {len(self.data)} point pairs → {self.out}")

    def load_pair(self):
        import cv2

        lp, rp = self.pairs[self.i]
        left, right = cv2.imread(lp), cv2.imread(rp)
        if left is None or right is None:
            sys.exit(f"ERROR: cannot read {lp if left is None else rp}")
        if left.shape != right.shape:
            print(f"[warn] {lp} and {rp} have different sizes")
        h, w = left.shape[:2]
        self.scale = min(DISP_W / w, DISP_H / h)
        size = (int(w * self.scale), int(h * self.scale))
        self.left_w = size[0]
        self.canvas = np.hstack([cv2.resize(left, size), cv2.resize(right, size)])
        self.pending = {}
        print(f"\n=== pair {self.i + 1}/{len(self.pairs)}:  {lp}  <->  {rp}")

    def show(self):
        import cv2

        img = self.canvas.copy()
        h = img.shape[0]
        s = self.scale
        cv2.line(img, (self.left_w, 0), (self.left_w, h), (255, 255, 0), 2)
        for e in self.data:
            if e.get("left_file") != self.pairs[self.i][0]:
                continue
            lx, ly = int(e["left_pt"][0] * s), int(e["left_pt"][1] * s)
            rx, ry = int(e["right_pt"][0] * s) + self.left_w, int(e["right_pt"][1] * s)
            cv2.circle(img, (lx, ly), 5, (0, 255, 0), -1)
            cv2.circle(img, (rx, ry), 5, (0, 255, 0), -1)
            cv2.putText(img, f'{e["distance"]:.2f}m', (lx + 8, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if "left" in self.pending:
            lx, ly = (int(c * s) for c in self.pending["left"])
            cv2.circle(img, (lx, ly), 6, (0, 0, 255), -1)
        step = "click RIGHT image" if "left" in self.pending else "click LEFT image"
        cv2.putText(img, f"pair {self.i + 1}/{len(self.pairs)} | {step} | {len(self.data)} points",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW, img)

    def on_mouse(self, event, x, y, flags, _param):
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        s = self.scale
        if "left" not in self.pending:
            if x >= self.left_w:
                print("  click the LEFT image first")
                return
            self.pending["left"] = [x / s, y / s]
            print(f"  left  ({x / s:.1f}, {y / s:.1f}) — now click the same object in the RIGHT image")
        else:
            if x < self.left_w:
                print("  now click the RIGHT image")
                return
            right = [(x - self.left_w) / s, y / s]
            print(f"  right ({right[0]:.1f}, {right[1]:.1f})")
            dist = ask_distance()
            if dist is not None:
                lp, rp = self.pairs[self.i]
                self.data.append({"left_file": lp, "right_file": rp, "left_pt": self.pending["left"],
                                  "right_pt": right, "distance": dist})
                self.save()
            self.pending = {}
        self.show()

    def run(self):
        import cv2

        self.load_pair()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.show()
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n") and self.i < len(self.pairs) - 1:
                self.i += 1
                self.load_pair()
                self.show()
            elif k == ord("p") and self.i > 0:
                self.i -= 1
                self.load_pair()
                self.show()
            elif k == ord("u"):
                if self.pending:
                    self.pending = {}
                    print("  pending click undone")
                elif self.data:
                    rm = self.data.pop()
                    self.save()
                    print(f"  removed {rm['left_pt']} <-> {rm['right_pt']}")
                self.show()
        cv2.destroyAllWindows()
        print(f"Done: {len(self.data)} point pairs in {self.out}")


def ask_distance():
    while True:
        s = input("  distance to this object in metres (Enter = cancel): ").strip().replace(",", ".")
        if not s:
            print("  cancelled")
            return None
        try:
            d = float(s)
            if d > 0:
                return d
        except ValueError:
            pass
        print("  please enter a positive number")


def main(left_dir="left", right_dir="right", out="training_data.json"):
    pairs = build_pairs(left_dir, right_dir)
    if not pairs:
        sys.exit(f"No image pairs found — put images in {left_dir}/ and {right_dir}/")
    print(f"Found {len(pairs)} pair(s)")
    Collector(pairs, out).run()
