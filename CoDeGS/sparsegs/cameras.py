"""Camera utilities: pseudo (unseen) view sampling and spiral video paths. OpenCV convention."""
import numpy as np
import torch


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def look_at_w2c(position, target, up_hint):
    """OpenCV camera (x right, y down, z forward) at `position` looking at `target`."""
    z = _normalize(target - position)
    x = _normalize(np.cross(z, up_hint))  # right
    y = np.cross(z, x)                     # down
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = position
    return np.linalg.inv(c2w)


def average_pose(c2ws):
    center = c2ws[:, :3, 3].mean(0)
    z = _normalize(c2ws[:, :3, 2].sum(0))
    y_down = c2ws[:, :3, 1].sum(0)
    x = _normalize(np.cross(y_down, z))
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1)
    m[:3, 3] = center
    return m


def _slerp(q0, q1, t):
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def _rot2quat(R):
    from .colmap_io import rotmat2qvec
    return rotmat2qvec(R)


def _quat2rot(q):
    from .colmap_io import qvec2rotmat
    return qvec2rotmat(q)


class PseudoViewSampler:
    """Virtual training-free views: interpolate (slerp/lerp) between two random training cameras
    and add a small random translation; a common unseen-view regulariser in sparse-view work."""

    def __init__(self, train_w2cs, noise_frac=0.1, seed=0):
        self.c2ws = np.linalg.inv(train_w2cs)
        self.q = [_rot2quat(c[:3, :3]) for c in self.c2ws]
        centers = self.c2ws[:, :3, 3]
        dists = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        self.baseline = float(np.median(dists[dists > 0])) if (dists > 0).any() else 1.0
        self.noise = noise_frac * self.baseline
        self.rng = np.random.default_rng(seed)

    def sample(self):
        n = len(self.c2ws)
        i, j = self.rng.choice(n, 2, replace=n < 2)
        t = self.rng.uniform(0, 1)
        R = _quat2rot(_slerp(self.q[i], self.q[j], t))
        p = (1 - t) * self.c2ws[i, :3, 3] + t * self.c2ws[j, :3, 3] + self.rng.normal(0, self.noise, 3)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = p
        return np.linalg.inv(c2w)


def spiral_path(c2ws, focus_depth, n_frames=120, n_rots=2, rad_scale=1.0, zrate=0.5):
    """LLFF-style spiral around the average camera (forward-facing scenes)."""
    avg = average_pose(c2ws)
    local = (np.linalg.inv(avg) @ c2ws)[:, :3, 3]
    rads = np.percentile(np.abs(local), 90, 0) * rad_scale
    rads = np.maximum(rads, 1e-3)
    target = avg[:3, :3] @ np.array([0, 0, focus_depth]) + avg[:3, 3]
    up = -avg[:3, 1]
    out = []
    for th in np.linspace(0, 2 * np.pi * n_rots, n_frames, endpoint=False):
        off = np.array([np.cos(th), -np.sin(th), -np.sin(th * zrate)]) * rads
        pos = avg[:3, :3] @ off + avg[:3, 3]
        out.append(look_at_w2c(pos, target, up))
    return np.stack(out)


def to_torch(m, device):
    return torch.as_tensor(np.asarray(m), dtype=torch.float32, device=device)
