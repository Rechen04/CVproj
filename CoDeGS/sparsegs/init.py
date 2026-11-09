"""Gaussian initialisation: SfM (train views), random, or depth-prior back-projection."""
import math

import numpy as np
import torch

C0 = 0.28209479177387814


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def knn_mean_dist2(xyz, k=3):
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(xyz).query(xyz, k=k + 1)
        return (d[:, 1:] ** 2).mean(1)
    except ImportError:  # chunked torch fallback
        x = torch.from_numpy(xyz).float()
        out = []
        for s in range(0, len(x), 4096):
            d = torch.cdist(x[s:s + 4096], x)
            out.append(d.topk(k + 1, largest=False).values[:, 1:].pow(2).mean(1))
        return torch.cat(out).numpy()


def random_points(ref_xyz, n, rng, expand=1.2):
    lo, hi = np.percentile(ref_xyz, 2, 0), np.percentile(ref_xyz, 98, 0)
    c, h = (lo + hi) / 2, (hi - lo) / 2 * expand
    xyz = c + (rng.random((n, 3)) * 2 - 1) * h
    rgb = rng.random((n, 3))
    return xyz.astype(np.float32), rgb.astype(np.float32)


def make_splats(xyz, rgb, sh_degree=3, init_opacity=0.1, init_scale=1.0, device="cuda"):
    """Returns an nn.ParameterDict with gsplat-style keys."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    N = len(xyz)
    d2 = np.clip(knn_mean_dist2(xyz), 1e-10, None)
    scales = np.log(np.sqrt(d2) * init_scale)[:, None].repeat(3, 1)
    quats = np.zeros((N, 4), np.float32)
    quats[:, 0] = 1.0
    K = (sh_degree + 1) ** 2
    sh0 = rgb_to_sh(rgb)[:, None, :]
    shN = np.zeros((N, K - 1, 3), np.float32)
    logit = math.log(init_opacity / (1 - init_opacity))
    params = {
        "means": torch.from_numpy(xyz),
        "scales": torch.from_numpy(scales.astype(np.float32)),
        "quats": torch.from_numpy(quats),
        "opacities": torch.full((N,), logit),
        "sh0": torch.from_numpy(sh0.astype(np.float32)),
        "shN": torch.from_numpy(shN),
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().to(device)) for k, v in params.items()})


def voxel_downsample(xyz, rgb, voxel):
    if voxel <= 0 or len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]
