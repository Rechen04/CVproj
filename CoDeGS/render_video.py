"""Render spiral fly-through videos; with several checkpoints, produce a side-by-side comparison
(top: RGB, bottom: depth) with labels -- the main 'wow' asset for the slides.

python render_video.py --ckpts results/fern/3dgs_3v/ckpt.pt results/fern/full_3v/ckpt.pt \
       --labels "3DGS" "CoDe-GS (ours)" --out videos/fern_compare.mp4
"""
import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from sparsegs import cameras as camlib
from sparsegs.render import rasterize
from sparsegs.utils import colorize_depth


def load(ckpt, dev):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(dev)) for k, v in d["splats"].items()})
    return splats, d["meta"]


def label(img, text, scale=0.6):
    img = np.ascontiguousarray(img)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rad_scale", type=float, default=1.0, help="spiral radius multiplier (bigger = harder novel views)")
    ap.add_argument("--path_from", default="all", choices=["all", "train"],
                    help="poses defining the spiral radius: all cameras (wider, LLFF-style) or train only")
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--backend", default="gsplat", choices=["gsplat", "torch"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    models = [load(c, dev) for c in a.ckpts]
    labels = a.labels or [os.path.basename(os.path.dirname(c)) for c in a.ckpts]
    meta = models[0][1]
    H, W = int(meta["H"]), int(meta["W"])
    w2cs = meta["w2cs"] if a.path_from == "all" else meta["w2cs"][meta["train_ids"]]
    path = camlib.spiral_path(np.linalg.inv(w2cs), meta["focus"], n_frames=a.frames, rad_scale=a.rad_scale)
    K = torch.from_numpy(meta["Ks"][meta["train_ids"][0]]).float().to(dev)[None]
    bg = torch.full((1, 3), float(meta.get("bg", 0.0)), device=dev)

    lohi = None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    for fi, w2c in enumerate(path):
        vm = torch.from_numpy(w2c).float().to(dev)[None]
        tops, bots = [], []
        for (splats, m), lab in zip(models, labels):
            with torch.no_grad():
                r, al, _ = rasterize(splats, vm, K, W, H, int(m["sh_degree"]), a.backend, "RGB+ED", backgrounds=bg)
            rgb = r[0, ..., :3].clamp(0, 1).cpu().numpy()
            dep = r[0, ..., 3].cpu().numpy()
            mask = al[0, ..., 0].cpu().numpy() > 0.5
            if lohi is None:  # fix colour range from the first model / first frame for comparability
                disp = 1 / np.clip(dep[mask] if mask.any() else dep.reshape(-1), 1e-6, None)
                lohi = (np.percentile(disp, 1), np.percentile(disp, 99))
            rgb8 = (rgb * 255).astype(np.uint8)
            if a.upscale > 1:
                rgb8 = cv2.resize(rgb8, (W * a.upscale, H * a.upscale), interpolation=cv2.INTER_CUBIC)
            tops.append(label(rgb8, lab))
            if not a.no_depth:
                d8 = (colorize_depth(dep, *lohi, mask=mask) * 255).astype(np.uint8)
                if a.upscale > 1:
                    d8 = cv2.resize(d8, (W * a.upscale, H * a.upscale), interpolation=cv2.INTER_NEAREST)
                bots.append(d8)
        frame = np.concatenate(tops, 1)
        if bots:
            frame = np.concatenate([frame, np.concatenate(bots, 1)], 0)
        if fi == 0:
            imageio.imwrite(os.path.splitext(a.out)[0] + "_frame0.png", frame)
        writer.append_data(frame)
    writer.close()
    print("saved", a.out)


if __name__ == "__main__":
    main()
