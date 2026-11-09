"""Scene loading for LLFF / Mip-NeRF360 style COLMAP folders, with the standard sparse-view split.

Sparse-view protocol (RegNeRF / FreeNeRF / DNGaussian / FSGS convention):
  * test  = every `hold`-th image (index % 8 == 0) in name-sorted order
  * train = from the remaining images pick `n_views` with np.linspace (rounded)
  * resolution: images_8 for LLFF (~504x378)
All poses here are OpenCV convention world->camera (x right, y down, z forward).
"""
import os
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np

from . import colmap_io

IMG_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


@dataclass
class Scene:
    name: str
    images: np.ndarray        # [N,H,W,3] float32 in [0,1]
    Ks: np.ndarray            # [N,3,3]
    w2cs: np.ndarray          # [N,4,4]
    names: list
    train_ids: np.ndarray
    test_ids: np.ndarray
    pts_xyz: np.ndarray       # all SfM points [M,3]
    pts_rgb: np.ndarray       # [M,3] uint8
    pts_tracks: list          # list of arrays with *scene indices* (not colmap ids) observing each point

    @property
    def H(self):
        return self.images.shape[1]

    @property
    def W(self):
        return self.images.shape[2]

    def c2ws(self):
        return np.linalg.inv(self.w2cs)

    def train_sfm_points(self, min_views=2):
        """SfM points observed by >= min_views TRAINING views.

        Note: point positions come from the full-view COLMAP reconstruction (standard LLFF
        release); we only keep points that the training views alone could triangulate.
        This is the usual approximation; state it in the report.
        """
        tr = set(self.train_ids.tolist())
        keep = np.array([len(tr.intersection(t.tolist())) >= min_views for t in self.pts_tracks], dtype=bool)
        return self.pts_xyz[keep], self.pts_rgb[keep]


def _list_images(d):
    return sorted([f for f in os.listdir(d) if f.endswith(IMG_EXT)])


def load_scene(root, factor=8, n_views=3, hold=8, max_images=None):
    name = os.path.basename(os.path.normpath(root))
    sp = os.path.join(root, "sparse", "0")
    if not os.path.isdir(sp):
        sp = os.path.join(root, "sparse")
    cams = colmap_io.read_cameras_binary(os.path.join(sp, "cameras.bin"))
    imgs = colmap_io.read_images_binary(os.path.join(sp, "images.bin"))
    xyz, rgb, _pids, tracks = colmap_io.read_points3D_binary(os.path.join(sp, "points3D.bin"))

    colmap_sorted = sorted(imgs.values(), key=lambda im: im.name)
    img_dir = os.path.join(root, f"images_{factor}") if factor > 1 else os.path.join(root, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"{img_dir} not found (LLFF ships images_4 / images_8).")
    files = _list_images(img_dir)
    if len(files) != len(colmap_sorted):
        raise RuntimeError(f"{img_dir}: {len(files)} files vs {len(colmap_sorted)} COLMAP images")
    if max_images:
        colmap_sorted, files = colmap_sorted[:max_images], files[:max_images]

    images, Ks, w2cs, names = [], [], [], []
    for im, fn in zip(colmap_sorted, files):
        arr = imageio.imread(os.path.join(img_dir, fn))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = arr[..., :3].astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        cam = cams[im.camera_id]
        fx, fy, cx, cy = cam.intrinsics()
        sx, sy = w / cam.width, h / cam.height
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float64)
        images.append(arr)
        Ks.append(K)
        w2cs.append(im.w2c())
        names.append(im.name)
    shapes = {a.shape for a in images}
    if len(shapes) != 1:
        raise RuntimeError(f"inconsistent image sizes {shapes}")
    dist_models = {c.model for c in cams.values() if c.model not in ("PINHOLE", "SIMPLE_PINHOLE")}
    if dist_models:
        print(f"  note: COLMAP camera model {dist_models} -> using f/cx/cy only, radial distortion ignored "
              f"(same as the NeRF/RegNeRF/DNGaussian LLFF protocol; LLFF ships SIMPLE_RADIAL)")

    N = len(images)
    all_ids = np.arange(N)
    test_ids = all_ids[all_ids % hold == 0]
    cand = all_ids[all_ids % hold != 0]
    if n_views and n_views < len(cand):
        sub = np.round(np.linspace(0, len(cand) - 1, n_views)).astype(int)
        train_ids = cand[sub]
    else:
        train_ids = cand

    # colmap image id -> scene index
    id2idx = {im.id: i for i, im in enumerate(colmap_sorted)}
    tracks_idx = [np.array([id2idx[t] for t in tr if t in id2idx], dtype=np.int64) for tr in tracks]

    return Scene(name, np.stack(images), np.stack(Ks), np.stack(w2cs), names,
                 train_ids, test_ids, xyz, rgb, tracks_idx)


def camera_scene_scale(w2cs):
    """gsplat / 3DGS 'scene extent': 1.1 * max distance of camera centres from their mean."""
    c2w = np.linalg.inv(w2cs)
    centers = c2w[:, :3, 3]
    d = np.linalg.norm(centers - centers.mean(0), axis=1)
    return float(max(d.max(), 1e-6) * 1.1)
