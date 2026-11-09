"""Generate a tiny synthetic LLFF-format scene (COLMAP sparse/0 + images_8) for pipeline smoke tests.

Scene: checkerboard back wall + floor + 3 coloured spheres, forward-facing camera grid.
Usage: python tests/make_synthetic_scene.py --out data/synthetic/toy --n 16 --w 64 --h 48
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparsegs import colmap_io  # noqa: E402

SPHERES = [(np.array([-0.8, 0.2, 3.0]), 0.6, np.array([0.9, 0.2, 0.2])),
           (np.array([0.7, 0.4, 3.6]), 0.7, np.array([0.2, 0.8, 0.3])),
           (np.array([0.0, -0.5, 4.2]), 0.5, np.array([0.2, 0.3, 0.9]))]
WALL_Z, FLOOR_Y = 6.0, 1.0


def shade(points, kind, idx):
    if kind == "sphere":
        c, r, col = SPHERES[idx]
        n = (points - c) / r
        lam = np.clip(n @ np.array([-0.3, -0.6, -0.7]), 0.15, 1)[:, None]
        stripes = 0.85 + 0.15 * np.sign(np.sin(points[:, 1:2] * 12))
        return col * lam * stripes
    if kind == "wall":
        chk = (np.floor(points[:, 0] * 2) + np.floor(points[:, 1] * 2)) % 2
        return np.where(chk[:, None] > 0, [0.85, 0.8, 0.6], [0.35, 0.3, 0.45])
    chk = (np.floor(points[:, 0] * 2) + np.floor(points[:, 2] * 2)) % 2
    return np.where(chk[:, None] > 0, [0.6, 0.6, 0.6], [0.25, 0.25, 0.25])


def trace(origin, dirs):
    n = len(dirs)
    t_best = np.full(n, np.inf)
    kind = np.array(["none"] * n, dtype=object)
    sid = np.zeros(n, int)
    for k, (c, r, _) in enumerate(SPHERES):
        oc = origin - c
        b = dirs @ oc
        cc = oc @ oc - r * r
        disc = b * b - cc
        t = -b - np.sqrt(np.clip(disc, 0, None))
        hit = (disc > 0) & (t > 0) & (t < t_best)
        t_best[hit], kind[hit], sid[hit] = t[hit], "sphere", k
    t = (WALL_Z - origin[2]) / dirs[:, 2]
    hit = (t > 0) & (t < t_best)
    t_best[hit], kind[hit] = t[hit], "wall"
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (FLOOR_Y - origin[1]) / dirs[:, 1]
    hit = (t > 0) & (t < t_best)
    t_best[hit], kind[hit] = t[hit], "floor"
    pts = origin + dirs * t_best[:, None]
    col = np.zeros((n, 3))
    for kd in ("wall", "floor"):
        m = kind == kd
        col[m] = shade(pts[m], kd, 0)
    for k in range(len(SPHERES)):
        m = (kind == "sphere") & (sid == k)
        col[m] = shade(pts[m], "sphere", k)
    return col, pts, t_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/toy")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--h", type=int, default=48)
    a = ap.parse_args()
    factor = 8
    FW, FH = a.w * factor, a.h * factor
    f = 0.9 * FW
    os.makedirs(os.path.join(a.out, "sparse", "0"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "images_8"), exist_ok=True)
    cam = colmap_io.Camera(1, "PINHOLE", FW, FH, np.array([f, f, FW / 2, FH / 2]))
    colmap_io.write_cameras_binary(os.path.join(a.out, "sparse", "0", "cameras.bin"), [cam])

    gx = int(np.ceil(np.sqrt(a.n)))
    imgs, all_pts, all_cols = [], [], []
    K8 = np.array([[f / factor, 0, a.w / 2], [0, f / factor, a.h / 2], [0, 0, 1]])
    for k in range(a.n):
        cx, cy = (k % gx) / max(gx - 1, 1) - 0.5, (k // gx) / max(gx - 1, 1) - 0.5
        pos = np.array([cx * 1.2, cy * 0.6, 0.0])
        target = np.array([0, 0, 4.0])
        z = (target - pos) / np.linalg.norm(target - pos)
        x = np.cross(z, [0, -1, 0]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c2w = np.eye(4); c2w[:3, :3] = np.stack([x, y, z], 1); c2w[:3, 3] = pos
        w2c = np.linalg.inv(c2w)
        vs, us = np.mgrid[0:a.h, 0:a.w] + 0.5
        rays = np.stack([(us - K8[0, 2]) / K8[0, 0], (vs - K8[1, 2]) / K8[1, 1], np.ones_like(us)], -1).reshape(-1, 3)
        dirs = rays @ c2w[:3, :3].T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        col, pts, _ = trace(pos, dirs)
        img = col.reshape(a.h, a.w, 3)
        name = f"IMG_{k:04d}.JPG"
        imageio.imwrite(os.path.join(a.out, "images_8", f"image{k:03d}.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))
        sel = np.random.default_rng(k).choice(len(pts), 40, replace=False)
        all_pts.append(pts[sel]); all_cols.append(col[sel])
        imgs.append(colmap_io.Image(k + 1, colmap_io.rotmat2qvec(w2c[:3, :3]), w2c[:3, 3], 1, name))
    xyz = np.concatenate(all_pts)
    rgb = (np.clip(np.concatenate(all_cols), 0, 1) * 255).astype(np.uint8)
    # tracks: every image in which the point is in front & inside the frame (no occlusion test; toy only)
    tracks = []
    for p in xyz:
        tr = []
        for im in imgs:
            w2c = im.w2c(); pc = w2c[:3, :3] @ p + w2c[:3, 3]
            if pc[2] <= 0:
                continue
            u, v = (K8 @ (pc / pc[2]))[:2]
            if 0 <= u < a.w and 0 <= v < a.h:
                tr.append(im.id)
        tracks.append(tr)
    colmap_io.write_images_binary(os.path.join(a.out, "sparse", "0", "images.bin"), imgs)
    colmap_io.write_points3D_binary(os.path.join(a.out, "sparse", "0", "points3D.bin"), xyz, rgb, tracks)
    print(f"wrote {a.n} views {a.w}x{a.h}, {len(xyz)} points -> {a.out}")


if __name__ == "__main__":
    main()
