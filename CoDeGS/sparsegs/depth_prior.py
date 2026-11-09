"""Monocular depth prior (Depth Anything V2 via HF transformers, fully offline) + geometry helpers.

Key pieces of the method:
  1. predict relative disparity with DAv2 for every training view (cached to disk)
  2. align it to metric scale with the training-view SfM points (robust affine fit in disparity space)
  3. cross-view consistency confidence: a pixel's depth is trusted if, after back-projection,
     it agrees with the aligned depth of at least one other training view
"""
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MonoDepth:
    """Wraps transformers' DepthAnything; input [B,3,H,W] in [0,1] -> relative disparity [B,H,W]."""

    def __init__(self, model_dir, device="cuda", lower_bound=518, fp16=True):
        from transformers import AutoModelForDepthEstimation
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Depth model dir {model_dir} not found (see offline/README_OFFLINE.md)")
        self.model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.device, self.lb = device, lower_bound
        self.dtype = torch.float16 if (fp16 and str(device).startswith("cuda")) else torch.float32
        self.model = self.model.to(self.dtype)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _size(self, h, w):
        s = max(self.lb / h, self.lb / w)  # "lower_bound" resize like the DAv2 processor
        nh = max(int(np.ceil(h * s / 14) * 14), 14)
        nw = max(int(np.ceil(w * s / 14) * 14), 14)
        return nh, nw

    @torch.no_grad()
    def __call__(self, img):
        B, _, H, W = img.shape
        x = F.interpolate(img.to(self.device), size=self._size(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
        x = ((x - self.mean) / self.std).to(self.dtype)
        out = self.model(pixel_values=x).predicted_depth  # [B,h,w]
        out = F.interpolate(out[:, None].float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        return out


def compute_mono_disparities(scene, ids, model_dir, cache_dir, device="cuda", tag="dav2"):
    """Returns dict id -> relative disparity [H,W] (np.float32), cached per image."""
    os.makedirs(cache_dir, exist_ok=True)
    out, todo = {}, []
    for i in ids:
        p = os.path.join(cache_dir, f"{tag}_{os.path.splitext(scene.names[i])[0]}_{scene.W}x{scene.H}.npy")
        if os.path.exists(p):
            out[int(i)] = np.load(p)
        else:
            todo.append((int(i), p))
    if todo:
        model = MonoDepth(model_dir, device=device)
        for i, p in todo:
            img = torch.from_numpy(scene.images[i]).permute(2, 0, 1)[None].float()
            d = model(img)[0].cpu().numpy().astype(np.float32)
            np.save(p, d)
            out[i] = d
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def project(xyz, K, w2c):
    """xyz [P,3] world -> uv [P,2] pixels, z [P] camera depth (numpy)."""
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = (cam[:, :2] / np.clip(z[:, None], 1e-8, None)) @ K[:2, :2].T + K[:2, 2]
    return uv, z


def align_disparity(mono, uv, z, n_iter=5, keep=0.8):
    """Robust least squares: 1/z ~= a * mono(uv) + b. Returns (a, b, n_inliers, corr)."""
    H, W = mono.shape
    m = (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1) & (z > 1e-6)
    uv, z = uv[m], z[m]
    if len(z) < 10:
        return None
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    x = mono[v, u].astype(np.float64)
    y = 1.0 / z
    corr = float(np.corrcoef(x, y)[0, 1]) if len(z) > 2 else 0.0
    idx = np.arange(len(x))
    a = b = None
    for _ in range(n_iter):
        A = np.stack([x[idx], np.ones(len(idx))], 1)
        a, b = np.linalg.lstsq(A, y[idx], rcond=None)[0]
        r = np.abs(a * x + b - y)
        idx = np.argsort(r)[: max(int(keep * len(x)), 10)]
    return float(a), float(b), len(idx), corr


def aligned_depths(scene, mono, min_views=2):
    """Metric depth maps for training views using SfM points seen by >= min_views train views.

    Returns depth dict id -> [H,W] float32 and a per-view info dict (scale, shift, #pts, corr).
    """
    xyz, _ = scene.train_sfm_points(min_views=min_views)
    if len(xyz) < 30:  # fall back to any SfM point (still only used for scale)
        xyz = scene.pts_xyz
        warnings.warn("few train-view SfM points; aligning with all SfM points")
    depths, info = {}, {}
    for i, d in mono.items():
        uv, z = project(xyz, scene.Ks[i], scene.w2cs[i])
        zpos = z[z > 0]
        far = np.percentile(zpos, 99) * 3 if len(zpos) else 100.0
        near = np.percentile(zpos, 1) * 0.3 if len(zpos) else 0.01
        res = align_disparity(d, uv, z)
        if res is None:
            raise RuntimeError(f"cannot align mono depth for view {scene.names[i]}")
        a, b, n, corr = res
        if a <= 0 or corr < 0:
            warnings.warn(f"view {scene.names[i]}: negative correlation ({corr:.2f}) between mono output and SfM disparity")
        disp = np.clip(a * d + b, 1.0 / far, 1.0 / near)
        depths[i] = (1.0 / disp).astype(np.float32)
        info[i] = dict(scale=a, shift=b, n_pts=n, corr=corr)
    return depths, info


@torch.no_grad()
def consistency_confidence(depths, Ks, w2cs, ids, tau=0.1, floor=0.2, device="cpu"):
    """Cross-view depth consistency.

    For pixel p in view i with aligned depth D_i(p): back-project to X, project into view j,
    compare z_j(X) with D_j(pi_j(X)). conf_ij = exp(-(rel_err/tau)^2).
    conf_i = max_j conf_ij; pixels unseen by any other view get `floor`.
    Returns dict id -> conf [H,W] in [floor, 1], plus coverage stats.
    """
    ids = [int(i) for i in ids]
    D = {i: torch.from_numpy(depths[i]).float().to(device) for i in ids}
    H, W = D[ids[0]].shape
    vs, us = torch.meshgrid(torch.arange(H, device=device).float(), torch.arange(W, device=device).float(), indexing="ij")
    pix = torch.stack([us, vs, torch.ones_like(us)], -1).reshape(-1, 3)
    confs, stats = {}, {}
    for i in ids:
        Ki = torch.from_numpy(Ks[i]).float().to(device)
        c2w = torch.from_numpy(np.linalg.inv(w2cs[i])).float().to(device)
        rays = pix @ torch.linalg.inv(Ki).T
        Xc = rays * D[i].reshape(-1, 1)
        Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
        best = torch.zeros(H * W, device=device)
        seen = torch.zeros(H * W, dtype=torch.bool, device=device)
        for j in ids:
            if j == i:
                continue
            Kj = torch.from_numpy(Ks[j]).float().to(device)
            w2c = torch.from_numpy(w2cs[j]).float().to(device)
            Xj = Xw @ w2c[:3, :3].T + w2c[:3, 3]
            z = Xj[:, 2]
            uv = (Xj[:, :2] / z.clamp_min(1e-6)[:, None]) @ Kj[:2, :2].T + Kj[:2, 2]
            inside = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            grid = torch.stack([uv[:, 0] / (W - 1) * 2 - 1, uv[:, 1] / (H - 1) * 2 - 1], -1).view(1, 1, -1, 2)
            Dj = F.grid_sample(D[j][None, None], grid, mode="nearest", align_corners=True).view(-1)
            # occlusion-aware: if X is behind the surface seen by j, j cannot judge it -> skip
            rel = (z - Dj) / z.clamp_min(1e-6)
            visible = inside & (rel < 3 * tau)
            c = torch.exp(-(rel.abs() / tau) ** 2)
            best = torch.where(visible, torch.maximum(best, c), best)
            seen |= visible
        conf = torch.where(seen, best, torch.zeros_like(best)).clamp_min(floor)
        confs[i] = conf.view(H, W).cpu().numpy().astype(np.float32)
        stats[i] = dict(seen=float(seen.float().mean()), high=float((conf > 0.5).float().mean()))
    return confs, stats


def backproject(depth, K, w2c, rgb, stride=2, mask=None):
    """Depth map -> world points/colours (numpy)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[vs, us]
    keep = np.isfinite(d) & (d > 0)
    if mask is not None:
        keep &= mask[vs, us]
    us, vs, d = us[keep], vs[keep], d[keep]
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(d)], -1)
    Xc = rays * d[:, None]
    c2w = np.linalg.inv(w2c)
    Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
    return Xw.astype(np.float32), rgb[vs, us].astype(np.float32)
