"""Losses: photometric (L1 + D-SSIM), weighted global / patch Pearson depth correlation, smoothness."""
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.exp(-((torch.arange(size, device=device).float() - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(img1, img2, size=11):
    """img [B,3,H,W] in [0,1]; same definition as the 3DGS repo (gaussian window, sigma 1.5)."""
    C = img1.shape[1]
    w = _gauss_window(size, device=img1.device).expand(C, 1, size, size)
    mu1 = F.conv2d(img1, w, padding=size // 2, groups=C)
    mu2 = F.conv2d(img2, w, padding=size // 2, groups=C)
    s11 = F.conv2d(img1 * img1, w, padding=size // 2, groups=C) - mu1 ** 2
    s22 = F.conv2d(img2 * img2, w, padding=size // 2, groups=C) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, w, padding=size // 2, groups=C) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m.mean()


def photometric_loss(pred, gt, lambda_dssim=0.2):
    """pred/gt [H,W,3]."""
    l1 = (pred - gt).abs().mean()
    s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - s), l1


def weighted_pearson(x, y, w, eps=1e-8):
    """Pearson correlation over last dim with non-negative weights. Returns corr and weight mass."""
    ws = w.sum(-1, keepdim=True).clamp_min(eps)
    mx = (w * x).sum(-1, keepdim=True) / ws
    my = (w * y).sum(-1, keepdim=True) / ws
    dx, dy = x - mx, y - my
    cov = (w * dx * dy).sum(-1) / ws[..., 0]
    vx = (w * dx * dx).sum(-1) / ws[..., 0]
    vy = (w * dy * dy).sum(-1) / ws[..., 0]
    return cov / torch.sqrt(vx * vy + eps), ws[..., 0], vx, vy


def depth_corr_loss(render_disp, mono_disp, weight, patch=32, w_global=1.0, w_local=1.0, min_mass=0.3):
    """Scale/shift-invariant depth prior.

    global term: 1 - weighted Pearson over the full image
    local  term: mean over random-offset non-overlapping patches of (1 - weighted Pearson)
    (inputs [H,W]; weight [H,W] >= 0, already including the alpha>0.5 mask and any confidence)
    """
    H, W = render_disp.shape
    loss = render_disp.new_zeros(())
    if w_global > 0:
        c, _, _, _ = weighted_pearson(render_disp.reshape(1, -1), mono_disp.reshape(1, -1), weight.reshape(1, -1))
        loss = loss + w_global * (1 - c.mean())
    if w_local > 0 and H >= patch and W >= patch:
        oy = int(torch.randint(0, patch, (1,))) if H >= 2 * patch else 0
        ox = int(torch.randint(0, patch, (1,))) if W >= 2 * patch else 0
        hh, ww = (H - oy) // patch * patch, (W - ox) // patch * patch

        def tiles(t):
            t = t[oy:oy + hh, ox:ox + ww]
            return t.reshape(hh // patch, patch, ww // patch, patch).permute(0, 2, 1, 3).reshape(-1, patch * patch)

        x, y, w = tiles(render_disp), tiles(mono_disp), tiles(weight)
        c, mass, vx, vy = weighted_pearson(x, y, w)
        # a patch counts if enough (weighted) pixels are valid and the prior is not flat there
        valid = (w.mean(-1) > min_mass * w.max().clamp_min(1e-8)) & (vy > 1e-10)
        if valid.any():
            loss = loss + w_local * (1 - c[valid]).mean()
    return loss


def edge_aware_smoothness(depth, img):
    """depth [H,W], img [H,W,3]."""
    dx = (depth[:, 1:] - depth[:, :-1]).abs()
    dy = (depth[1:, :] - depth[:-1, :]).abs()
    ix = (img[:, 1:] - img[:, :-1]).abs().mean(-1)
    iy = (img[1:, :] - img[:-1, :]).abs().mean(-1)
    return (dx * torch.exp(-10 * ix)).mean() + (dy * torch.exp(-10 * iy)).mean()


def psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-12)))
