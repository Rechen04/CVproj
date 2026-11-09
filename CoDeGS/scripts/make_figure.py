"""Qualitative comparison grid for slides: GT | method A | method B | ... (RGB row + depth row),
with per-image PSNR, from the test renders that train.py already saved.

python scripts/make_figure.py --scene_dir results/fern --runs 3dgs_3v depth_sfm_3v full_3v \
       --labels "3DGS" "Depth-reg." "Ours" --out figures/fern.png
(--view defaults to the first test view; --crop x0 y0 w h adds a zoomed inset row)
"""
import argparse
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np


def put(img, text):
    img = np.ascontiguousarray(img).copy()
    s = max(img.shape[1] / 600, 0.45)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--view", default=None)
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = a.labels or a.runs
    tdir = os.path.join(a.scene_dir, a.runs[0], "test")
    views = sorted(f[:-7] for f in os.listdir(tdir) if f.endswith("_gt.png"))
    view = a.view or views[0]
    gt = imageio.imread(os.path.join(tdir, f"{view}_gt.png"))[..., :3]
    row_crop = []
    gt_box = gt.copy()
    if a.crop:
        x, y, w, h = a.crop
        z = lambda im: cv2.resize(im[y:y + h, x:x + w], (gt.shape[1], int(gt.shape[1] * h / w)), interpolation=cv2.INTER_CUBIC)  # noqa
        row_crop.append(z(gt))
        cv2.rectangle(gt_box, (x, y), (x + w, y + h), (255, 255, 0), 2)
    row_rgb, row_dep = [put(gt_box, "Ground truth")], [np.zeros_like(gt)]
    for run, lab in zip(a.runs, labels):
        d = os.path.join(a.scene_dir, run)
        met = json.load(open(os.path.join(d, "metrics.json")))
        pv = {os.path.splitext(k)[0]: v for k, v in met["per_view"].items()}
        rgb = np.ascontiguousarray(imageio.imread(os.path.join(d, "test", f"{view}.png"))[..., :3])
        dep = imageio.imread(os.path.join(d, "test", f"{view}_depth.png"))[..., :3]
        if a.crop:
            row_crop.append(z(rgb))
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
        row_rgb.append(put(rgb, f"{lab}  {pv[view]['psnr']:.2f} dB"))
        row_dep.append(dep)
    rows = [np.concatenate(row_rgb, 1), np.concatenate(row_dep, 1)]
    if row_crop:
        rows.insert(1, np.concatenate(row_crop, 1))
    grid = np.concatenate(rows, 0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.imwrite(a.out, grid)
    print("saved", a.out, grid.shape)


if __name__ == "__main__":
    main()
