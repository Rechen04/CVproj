"""Misc: colormap, image saving, metrics wrapper (LPIPS offline-safe), PLY export."""
import json
import os
import warnings

import imageio.v2 as imageio
import numpy as np
import torch

from .losses import psnr, ssim

# Google "turbo" colormap, 256x3 polynomial approximation (no matplotlib dependency)
_kr = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
_kg = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
_kb = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])


def turbo(x):
    x = np.clip(x, 0, 1)[..., None] ** np.arange(6)
    rgb = np.stack([(x * _kr).sum(-1), (x * _kg).sum(-1), (x * _kb).sum(-1)], -1)
    return np.clip(rgb, 0, 1)


def colorize_depth(depth, lo=None, hi=None, mask=None):
    """Depth -> RGB using inverse depth (closer = warmer). Fixed lo/hi keep videos/figures comparable."""
    disp = 1.0 / np.clip(depth, 1e-6, None)
    v = disp[mask] if mask is not None and mask.any() else disp.reshape(-1)
    lo = np.percentile(v, 1) if lo is None else lo
    hi = np.percentile(v, 99) if hi is None else hi
    out = turbo((disp - lo) / max(hi - lo, 1e-8))
    if mask is not None:
        out[~mask] = 0
    return out


def save_img(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, (np.clip(arr, 0, 1) * 255).astype(np.uint8))


class Metrics:
    """PSNR / SSIM / LPIPS(vgg & alex). LPIPS needs torchvision weights in the torch hub cache
    (offline/README_OFFLINE.md); if missing it is skipped with a warning instead of crashing."""

    def __init__(self, device):
        self.device = device
        self.lpips = {}
        for net in ("vgg", "alex"):
            try:
                import lpips
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.lpips[net] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
            except Exception as e:  # noqa
                print(f"[metrics] LPIPS-{net} unavailable ({type(e).__name__}: {str(e)[:120]}); skipping")

    @torch.no_grad()
    def __call__(self, pred, gt):
        """pred/gt [H,W,3] in [0,1]"""
        pred, gt = pred.clamp(0, 1), gt.clamp(0, 1)
        out = dict(psnr=psnr(pred, gt), ssim=float(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        for net, fn in self.lpips.items():
            if min(pred.shape[:2]) < 32:  # too small for the VGG/AlexNet pooling stack (toy tests only)
                continue
            out[f"lpips_{net}"] = float(fn(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1))
        return out


def export_ply(splats, path):
    """Standard 3DGS PLY (opens in SuperSplat / antimatter15 / any 3DGS viewer)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from gsplat import export_splats
        export_splats(means=splats["means"].detach().cpu(), scales=splats["scales"].detach().cpu(),
                      quats=splats["quats"].detach().cpu(), opacities=splats["opacities"].detach().cpu(),
                      sh0=splats["sh0"].detach().cpu(), shN=splats["shN"].detach().cpu(), format="ply", save_to=path)
        return
    except Exception as e:  # noqa
        print(f"[ply] gsplat exporter failed ({e}); using built-in writer")
    m = splats["means"].detach().cpu().numpy()
    N = len(m)
    f_dc = splats["sh0"].detach().cpu().numpy().reshape(N, -1)
    f_rest = splats["shN"].detach().cpu().transpose(1, 2).reshape(N, -1).numpy()
    op = splats["opacities"].detach().cpu().numpy()[:, None]
    sc = splats["scales"].detach().cpu().numpy()
    q = splats["quats"].detach().cpu().numpy()
    cols = [m, np.zeros_like(m), f_dc, f_rest, op, sc, q]
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(f_dc.shape[1])] + \
            [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] + \
            [f"rot_{i}" for i in range(4)]
    data = np.concatenate(cols, 1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N).encode())
        for n in names:
            f.write(f"property float {n}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
