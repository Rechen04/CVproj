"""Rasterization wrapper.

backend="gsplat": gsplat CUDA rasterizer (used on the A100).
backend="torch" : tiny dense pure-PyTorch rasterizer, ONLY for CPU smoke tests of the pipeline
                  (O(pixels x gaussians) memory; use with tiny images).
Both return (renders [C,H,W,3 or 4], alphas [C,H,W,1], info) with gsplat-compatible info keys
so gsplat.strategy.DefaultStrategy works with either.
"""
import torch
import torch.nn.functional as F


def splat_activations(splats):
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N,K,3]
    return means, quats, scales, opacities, colors


def rasterize(splats, viewmats, Ks, width, height, sh_degree, backend="gsplat",
              render_mode="RGB+ED", backgrounds=None, absgrad=False, near_plane=0.01):
    means, quats, scales, opacities, colors = splat_activations(splats)
    if backend == "gsplat":
        from gsplat import rasterization
        return rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height, sh_degree=sh_degree,
            packed=False, absgrad=absgrad, render_mode=render_mode, backgrounds=backgrounds,
            near_plane=near_plane, rasterize_mode="classic",
        )
    if backend == "torch":
        return _dense_torch(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                            sh_degree, render_mode, backgrounds, near_plane)
    raise ValueError(backend)


# ----------------------------------------------------------------------------------------------
def _quat_to_rot(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def _dense_torch(means, quats, scales, opacities, colors, viewmats, Ks, W, H, sh_degree,
                 render_mode, backgrounds, near_plane, eps2d=0.3):
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast
    C, N = viewmats.shape[0], means.shape[0]
    R = _quat_to_rot(quats)
    S = torch.diag_embed(scales)
    cov3 = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)  # [N,3,3]

    ys, xs = torch.meshgrid(torch.arange(H, device=means.device), torch.arange(W, device=means.device), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5  # pixel centres, like gsplat
    # pass 1: projection for all cameras (means2d must be ONE tensor that the raster pass reads,
    # so that info["means2d"].grad is populated for densification)
    geo = []
    m2d_list = []
    for c in range(C):
        Rw, t = viewmats[c, :3, :3], viewmats[c, :3, 3]
        K = Ks[c]
        Xc = means @ Rw.T + t
        x, y, z = Xc.unbind(-1)
        zc = z.clamp_min(1e-6)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        m2d_list.append(torch.stack([fx * x / zc + cx, fy * y / zc + cy], -1))
        J = torch.zeros(N, 2, 3, device=means.device)
        J[:, 0, 0] = fx / zc
        J[:, 0, 2] = -fx * x / zc ** 2
        J[:, 1, 1] = fy / zc
        J[:, 1, 2] = -fy * y / zc ** 2
        cov_c = Rw @ cov3 @ Rw.T
        cov2 = J @ cov_c @ J.transpose(-1, -2) + eps2d * torch.eye(2, device=means.device)
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
        conic = torch.stack([cov2[:, 1, 1], -cov2[:, 0, 1], cov2[:, 0, 0]], -1) / det.clamp_min(1e-12)[:, None]
        geo.append((Rw, t, z, det, conic, cov2))
    means2d = torch.stack(m2d_list)  # [C,N,2]

    radii_all, out_cols, out_alpha = [], [], []
    for c in range(C):
        Rw, t, z, det, conic, cov2 = geo[c]
        m2d = means2d[c]
        b = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1])
        lam = b + torch.sqrt((b * b - det).clamp_min(0.01))
        rad = torch.ceil(3 * torch.sqrt(lam)).detach()
        valid = (z > near_plane) & (det > 0)
        inside = (m2d[:, 0] + rad > 0) & (m2d[:, 0] - rad < W) & (m2d[:, 1] + rad > 0) & (m2d[:, 1] - rad < H)
        valid = (valid & inside).detach()
        radii = torch.where(valid, rad, torch.zeros_like(rad)).int()
        radii_all.append(torch.stack([radii, radii], -1))

        # colours from SH
        campos = -Rw.T @ t
        dirs = F.normalize(means - campos, dim=-1)
        K_sh = (sh_degree + 1) ** 2 if sh_degree is not None else 1
        bases = _eval_sh_bases_fast(K_sh, dirs)  # [N,K_sh]
        rgb = (bases[..., None] * colors[:, :K_sh]).sum(1)
        rgb = torch.clamp_min(rgb + 0.5, 0.0)

        order = torch.argsort(z.detach())
        d = pix[:, None, :] - m2d[order][None]  # [P,N,2]
        cn = conic[order]
        power = -0.5 * (cn[:, 0] * d[..., 0] ** 2 + cn[:, 2] * d[..., 1] ** 2) - cn[:, 1] * d[..., 0] * d[..., 1]
        alpha = (opacities[order] * torch.exp(power)).clamp_max(0.999)
        alpha = torch.where((power <= 0) & (alpha >= 1 / 255) & valid[order][None], alpha, torch.zeros_like(alpha))
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
        w = alpha * T  # [P,N]
        col = w @ rgb[order]
        acc = w.sum(1, keepdim=True)
        feats = [col]
        if "D" in render_mode:
            dep = w @ z[order][:, None]
            if "ED" in render_mode:
                dep = dep / acc.clamp_min(1e-10)
            feats.append(dep)
        out = torch.cat(feats, -1)
        if render_mode in ("D", "ED"):
            out = out[:, 3:]
        if backgrounds is not None and render_mode.startswith("RGB"):
            out = torch.cat([out[:, :3] + (1 - acc) * backgrounds[c], out[:, 3:]], -1)
        out_cols.append(out.view(H, W, -1))
        out_alpha.append(acc.view(H, W, 1))
    info = dict(means2d=means2d, radii=torch.stack(radii_all), width=W, height=H, n_cameras=C,
                gaussian_ids=None, depths=None)
    return torch.stack(out_cols), torch.stack(out_alpha), info
