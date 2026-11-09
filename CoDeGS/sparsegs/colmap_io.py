"""Minimal COLMAP binary reader (cameras.bin / images.bin / points3D.bin).

Only the fields needed here are kept. Format reference: COLMAP src/colmap/scene/reconstruction_io.cc
"""
import struct
from dataclasses import dataclass, field

import numpy as np

# model_id -> (name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self):
        """Return fx, fy, cx, cy (distortion ignored; LLFF is undistorted PINHOLE / SIMPLE_*)."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # w, x, y, z  (world -> camera rotation)
    tvec: np.ndarray
    camera_id: int
    name: str
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def w2c(self):
        m = np.eye(4)
        m[:3, :3] = qvec2rotmat(self.qvec)
        m[:3, 3] = self.tvec
        return m


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def _read(f, n, fmt):
    return struct.unpack("<" + fmt, f.read(n))


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            cid, mid, w, h = _read(f, 24, "iiQQ")
            name, npar = CAMERA_MODELS[mid]
            params = np.array(_read(f, 8 * npar, "d" * npar))
            cams[cid] = Camera(cid, name, int(w), int(h), params)
    return cams


def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            props = _read(f, 64, "idddddddi")
            iid, q, t, cid = props[0], np.array(props[1:5]), np.array(props[5:8]), props[8]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            (n2d,) = _read(f, 8, "Q")
            data = _read(f, 24 * n2d, "ddq" * n2d)
            p3d = np.array(data[2::3], dtype=np.int64)
            imgs[iid] = Image(iid, q, t, cid, name.decode("utf-8"), p3d)
    return imgs


def read_points3D_binary(path):
    """Returns xyz [M,3], rgb [M,3] uint8, ids [M], tracks list of np.ndarray(image_ids)."""
    xyzs, rgbs, ids, tracks = [], [], [], []
    with open(path, "rb") as f:
        (num,) = _read(f, 8, "Q")
        for _ in range(num):
            props = _read(f, 43, "QdddBBBd")
            ids.append(props[0])
            xyzs.append(props[1:4])
            rgbs.append(props[4:7])
            (tl,) = _read(f, 8, "Q")
            tr = _read(f, 8 * tl, "ii" * tl)
            tracks.append(np.array(tr[0::2], dtype=np.int64))
    return (np.array(xyzs, dtype=np.float64).reshape(-1, 3), np.array(rgbs, dtype=np.uint8).reshape(-1, 3),
            np.array(ids, dtype=np.int64), tracks)


# ---------- writers (used only by the synthetic test-scene generator) ----------

def write_cameras_binary(path, cams):
    inv = {v[0]: k for k, v in CAMERA_MODELS.items()}
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for c in cams:
            f.write(struct.pack("<iiQQ", c.id, inv[c.model], c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *c.params))


def write_images_binary(path, imgs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for im in imgs:
            f.write(struct.pack("<idddddddi", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im.point3D_ids)))
            for pid in im.point3D_ids:
                f.write(struct.pack("<ddq", 0.0, 0.0, int(pid)))


def write_points3D_binary(path, xyz, rgb, tracks):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c, tr) in enumerate(zip(xyz, rgb, tracks)):
            f.write(struct.pack("<QdddBBBd", i + 1, *p, *[int(v) for v in c], 0.0))
            f.write(struct.pack("<Q", len(tr)))
            for iid in tr:
                f.write(struct.pack("<ii", int(iid), 0))


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q
