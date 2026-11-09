"""Sparse-view 3D Gaussian Splatting: vanilla 3DGS baseline and CoDe-GS (ours).

CoDe-GS = Consistency-weighted Depth-prior Gaussian Splatting
  (1) DInit : dense initialisation by back-projecting scale-aligned Depth-Anything-V2 depth
  (2) CW-D  : local+global Pearson depth prior, weighted by cross-view depth-consistency confidence
  (3) PV    : pseudo-view depth regularisation on interpolated unseen cameras

Example:
  python train.py --data_dir data/nerf_llff_data/fern --out_dir results --preset full --n_views 3
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from sparsegs import cameras as camlib
from sparsegs import depth_prior as dp
from sparsegs.data import camera_scene_scale, load_scene
from sparsegs.init import make_splats, random_points, voxel_downsample
from sparsegs.losses import depth_corr_loss, photometric_loss
from sparsegs.render import rasterize
from sparsegs.utils import Metrics, colorize_depth, dump_json, export_ply, save_img

PRESETS = {
    # name           init      depth-loss  pseudo
    "3dgs":          ("sfm",    "none",     False),  # original paper, sparse input
    "3dgs_random":   ("random", "none",     False),
    "depth_sfm":     ("sfm",    "uniform",  False),  # prior-work style mono-depth Pearson (FSGS/SparseGS-like)
    "dinit":         ("depth",  "none",     False),
    "dinit_depth":   ("depth",  "uniform",  False),
    "dinit_conf":    ("depth",  "conf",     False),
    "full_noconf":   ("depth",  "uniform",  True),
    "full":          ("depth",  "conf",     True),   # CoDe-GS
}


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--preset", default="full", choices=list(PRESETS))
    p.add_argument("--tag", default="", help="suffix for the run folder (e.g. sweep values)")
    p.add_argument("--n_views", type=int, default=3)
    p.add_argument("--factor", type=int, default=8)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0, help="debug only")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="gsplat", choices=["gsplat", "torch"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--bg", type=float, default=0.0)
    # depth prior
    p.add_argument("--depth_model", default="weights/Depth-Anything-V2-Large-hf")
    p.add_argument("--pseudo_depth_model", default="", help="model for pseudo views (default: --depth_model)")
    p.add_argument("--depth_cache", default="cache/mono_depth")
    p.add_argument("--w_depth_global", type=float, default=0.1)
    p.add_argument("--w_depth_local", type=float, default=0.1)
    p.add_argument("--depth_patch", type=int, default=32)
    p.add_argument("--depth_start", type=int, default=0)
    p.add_argument("--depth_end", type=int, default=-1, help="-1 = until the end")
    p.add_argument("--conf_tau", type=float, default=0.1)
    p.add_argument("--conf_floor", type=float, default=0.2)
    p.add_argument("--w_pseudo", type=float, default=0.05)
    p.add_argument("--pseudo_start", type=int, default=1500)
    p.add_argument("--pseudo_every", type=int, default=1)
    p.add_argument("--pseudo_noise", type=float, default=0.1)
    # init
    p.add_argument("--init", default="", choices=["", "sfm", "random", "depth"], help="override preset")
    p.add_argument("--init_stride", type=int, default=2)
    p.add_argument("--init_max_points", type=int, default=150000)
    p.add_argument("--init_opacity", type=float, default=0.1)
    p.add_argument("--random_n", type=int, default=20000)
    # densification (gsplat DefaultStrategy)
    p.add_argument("--refine_start", type=int, default=500)
    p.add_argument("--refine_stop_frac", type=float, default=0.6)
    p.add_argument("--refine_every", type=int, default=100)
    p.add_argument("--grow_grad2d", type=float, default=0.0002)
    p.add_argument("--reset_every", type=int, default=0, help="0 = no opacity reset (recommended for sparse views)")
    p.add_argument("--absgrad", action="store_true")
    # outputs
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--no_ply", action="store_true")
    p.add_argument("--save_test_renders", type=int, default=1)
    return p.parse_args(argv)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_optimizers(splats, scene_scale, steps):
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opts = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / steps))
    return opts, sched


def main(argv=None):
    args = get_args(argv)
    seed_all(args.seed)
    init_mode, depth_mode, use_pseudo = PRESETS[args.preset]
    if args.init:
        init_mode = args.init
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.backend == "gsplat" and dev.type != "cuda":
        raise SystemExit("gsplat backend needs CUDA; use --backend torch --device cpu for smoke tests")

    scene = load_scene(args.data_dir, factor=args.factor, n_views=args.n_views, hold=args.hold,
                       max_images=args.max_images or None)
    run = f"{args.preset}_{args.n_views}v" + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(args.out_dir, scene.name, run)
    os.makedirs(out, exist_ok=True)
    dump_json(vars(args), os.path.join(out, "args.json"))
    H, W = scene.H, scene.W
    tr, te = scene.train_ids, scene.test_ids
    print(f"[{scene.name}] {len(scene.names)} images {W}x{H}; train {tr.tolist()} test {te.tolist()}; preset={args.preset}")

    gt = torch.from_numpy(scene.images).float().to(dev)
    Ks = torch.from_numpy(scene.Ks).float().to(dev)
    viewmats = torch.from_numpy(scene.w2cs).float().to(dev)
    bg = torch.full((1, 3), args.bg, device=dev)
    scene_scale = camera_scene_scale(scene.w2cs[tr])
    sfm_xyz, sfm_rgb = scene.train_sfm_points(min_views=2)
    if len(sfm_xyz) < 50:
        print(f"  warning: only {len(sfm_xyz)} train-view SfM points; using points with >=1 train view")
        sfm_xyz, sfm_rgb = scene.train_sfm_points(min_views=1)
    sfm_rgb = sfm_rgb.astype(np.float32) / 255.0

    # ------------------------------------------------------------------ depth prior
    need_prior = init_mode == "depth" or depth_mode != "none"
    mono, depths, conf, prior_info = {}, {}, {}, {}
    if need_prior:
        cache = os.path.join(args.depth_cache, scene.name)
        mono = dp.compute_mono_disparities(scene, tr, args.depth_model, cache, device=str(dev))
        depths, align = dp.aligned_depths(scene, mono)
        conf, cstats = dp.consistency_confidence(depths, scene.Ks, scene.w2cs, tr, tau=args.conf_tau,
                                                 floor=args.conf_floor, device=str(dev))
        prior_info = {scene.names[i]: dict(**align[i], **cstats[i]) for i in tr}
        print("  depth prior:", json.dumps(prior_info, default=lambda x: round(x, 4)))
        for i in tr:  # figures for the slides
            nm = os.path.splitext(scene.names[i])[0]
            save_img(os.path.join(out, "prior", f"{nm}_rgb.png"), scene.images[i])
            save_img(os.path.join(out, "prior", f"{nm}_mono_depth.png"), colorize_depth(depths[i]))
            save_img(os.path.join(out, "prior", f"{nm}_confidence.png"), np.repeat(conf[i][..., None], 3, -1))

    # ------------------------------------------------------------------ init
    rng = np.random.default_rng(args.seed)
    if init_mode == "sfm":
        xyz, rgb = sfm_xyz, sfm_rgb
    elif init_mode == "random":
        xyz, rgb = random_points(scene.pts_xyz, args.random_n, rng)
    else:
        P, C = [sfm_xyz], [sfm_rgb]
        for i in tr:
            p, c = dp.backproject(depths[i], scene.Ks[i], scene.w2cs[i], scene.images[i], stride=args.init_stride)
            P.append(p)
            C.append(c)
        xyz, rgb = np.concatenate(P), np.concatenate(C)
        if len(xyz) > args.init_max_points:
            sel = rng.choice(len(xyz), args.init_max_points, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]
    xyz, rgb = voxel_downsample(xyz, rgb, 0.0)
    splats = make_splats(xyz, rgb, sh_degree=args.sh_degree, init_opacity=args.init_opacity, device=dev)
    print(f"  init={init_mode}: {len(xyz)} gaussians, scene_scale={scene_scale:.4f}")

    # ------------------------------------------------------------------ strategy / optim
    from gsplat.strategy import DefaultStrategy
    refine_stop = int(args.refine_stop_frac * args.steps)
    strategy = DefaultStrategy(
        refine_start_iter=args.refine_start, refine_stop_iter=refine_stop, refine_every=args.refine_every,
        grow_grad2d=args.grow_grad2d, absgrad=args.absgrad,
        reset_every=args.reset_every if args.reset_every > 0 else 10 ** 9, verbose=False)
    optimizers, sched = build_optimizers(splats, scene_scale, args.steps)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    mono_t = {i: torch.from_numpy(mono[i]).float().to(dev) for i in mono}
    conf_t = {i: torch.from_numpy(conf[i]).float().to(dev) for i in conf}
    pseudo_model = None
    if use_pseudo:
        pseudo_model = dp.MonoDepth(args.pseudo_depth_model or args.depth_model, device=str(dev))
        sampler = camlib.PseudoViewSampler(scene.w2cs[tr], noise_frac=args.pseudo_noise, seed=args.seed)
    metrics = Metrics(dev)
    depth_end = args.steps if args.depth_end < 0 else args.depth_end

    def evaluate(save=False):
        res = {}
        with torch.no_grad():
            for i in te:
                r, a, _ = rasterize(splats, viewmats[i:i + 1], Ks[i:i + 1], W, H, args.sh_degree, args.backend,
                                    "RGB+ED", backgrounds=bg)
                img = r[0, ..., :3].clamp(0, 1)
                res[scene.names[i]] = metrics(img, gt[i])
                if save:
                    nm = os.path.splitext(scene.names[i])[0]
                    save_img(os.path.join(out, "test", f"{nm}.png"), img.cpu().numpy())
                    save_img(os.path.join(out, "test", f"{nm}_gt.png"), scene.images[i])
                    d = r[0, ..., 3].cpu().numpy()
                    save_img(os.path.join(out, "test", f"{nm}_depth.png"), colorize_depth(d, mask=a[0, ..., 0].cpu().numpy() > 0.5))
        mean = {k: float(np.mean([v[k] for v in res.values()])) for k in next(iter(res.values()))}
        return mean, res

    # ------------------------------------------------------------------ train
    log = open(os.path.join(out, "log.jsonl"), "w")
    curve = []
    t0 = time.time()
    peak_n = len(splats["means"])
    for step in range(args.steps):
        i = int(rng.choice(tr))
        sh_now = min(step // args.sh_every, args.sh_degree)
        renders, alphas, info = rasterize(splats, viewmats[i:i + 1], Ks[i:i + 1], W, H, sh_now, args.backend,
                                          "RGB+ED", backgrounds=bg, absgrad=args.absgrad)
        rgb, depth, acc = renders[0, ..., :3], renders[0, ..., 3], alphas[0, ..., 0]
        loss, l1 = photometric_loss(rgb, gt[i])
        logs = {"step": step, "l1": float(l1)}
        strategy.step_pre_backward(splats, optimizers, state, step, info)

        if depth_mode != "none" and args.depth_start <= step < depth_end:
            w = (acc.detach() > 0.5).float()
            if depth_mode == "conf":
                w = w * conf_t[i]
            ld = depth_corr_loss(1.0 / depth.clamp_min(1e-3), mono_t[i], w, patch=args.depth_patch,
                                 w_global=args.w_depth_global, w_local=args.w_depth_local)
            loss = loss + ld
            logs["depth"] = float(ld)

        if use_pseudo and step >= args.pseudo_start and step % args.pseudo_every == 0:
            pw2c = camlib.to_torch(sampler.sample(), dev)[None]
            r2, a2, _ = rasterize(splats, pw2c, Ks[i:i + 1], W, H, sh_now, args.backend, "RGB+ED", backgrounds=bg)
            with torch.no_grad():
                m2 = pseudo_model(r2[0, ..., :3].detach().clamp(0, 1).permute(2, 0, 1)[None])[0]
            w2 = (a2[0, ..., 0].detach() > 0.5).float()
            lp = depth_corr_loss(1.0 / r2[0, ..., 3].clamp_min(1e-3), m2, w2, patch=args.depth_patch,
                                 w_global=1.0, w_local=1.0)
            loss = loss + args.w_pseudo * lp
            logs["pseudo"] = float(lp)

        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if args.reset_every > 0 and 0 < step < refine_stop and step % args.reset_every == 0:
            from gsplat.strategy.ops import reset_opa  # gsplat 1.5.3 DefaultStrategy never resets (precedence bug)
            reset_opa(splats, optimizers, state, value=strategy.prune_opa * 2.0)
        peak_n = max(peak_n, len(splats["means"]))

        if step % args.log_every == 0 or step == args.steps - 1:
            logs.update(loss=float(loss), n=len(splats["means"]), t=round(time.time() - t0, 1))
            log.write(json.dumps(logs) + "\n")
            log.flush()
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            m, _ = evaluate()
            curve.append(dict(step=step + 1, **m))
            print(f"  step {step + 1}: test PSNR {m['psnr']:.2f}  n={len(splats['means'])}")
    train_time = time.time() - t0
    log.close()

    final, per_view = evaluate(save=bool(args.save_test_renders))
    curve.append(dict(step=args.steps, **final))
    # render speed (gsplat only)
    fps = None
    if args.backend == "gsplat":
        torch.cuda.synchronize()
        ts = time.time()
        with torch.no_grad():
            for _ in range(50):
                rasterize(splats, viewmats[te[:1]], Ks[te[:1]], W, H, args.sh_degree, args.backend, "RGB")
        torch.cuda.synchronize()
        fps = 50 / (time.time() - ts)

    # focus depth for the video path: median depth of train SfM points in the mean camera
    avg = camlib.average_pose(np.linalg.inv(scene.w2cs[tr]))
    zc = ((sfm_xyz - avg[:3, 3]) @ avg[:3, :3])[:, 2]
    focus = float(np.median(zc[zc > 0])) if (zc > 0).any() else 1.0
    summary = dict(scene=scene.name, preset=args.preset, tag=args.tag, n_views=args.n_views, **final,
                   n_gaussians=len(splats["means"]), peak_gaussians=peak_n, train_sec=round(train_time, 1),
                   fps=fps, per_view=per_view, curve=curve, prior=prior_info, init_points=int(len(xyz)))
    torch.save(dict(splats={k: v.detach().cpu() for k, v in splats.items()},
                    meta=dict(Ks=scene.Ks, w2cs=scene.w2cs, H=H, W=W, train_ids=tr, test_ids=te,
                              names=scene.names, focus=focus, sh_degree=args.sh_degree, bg=args.bg,
                              sfm_xyz=sfm_xyz)), os.path.join(out, "ckpt.pt"))
    if not args.no_ply:
        export_ply(splats, os.path.join(out, "point_cloud.ply"))
    # Publish the completion marker after all deliverables have been saved.
    metrics_tmp = os.path.join(out, "metrics.json.tmp")
    dump_json(summary, metrics_tmp)
    os.replace(metrics_tmp, os.path.join(out, "metrics.json"))
    print(f"[{scene.name}/{run}] PSNR {final['psnr']:.2f} SSIM {final['ssim']:.3f} "
          + (f"LPIPS(vgg) {final['lpips_vgg']:.3f} " if 'lpips_vgg' in final else "")
          + f"| {len(splats['means'])} GS | {train_time:.0f}s" + (f" | {fps:.0f} FPS" if fps else ""))
    return summary


if __name__ == "__main__":
    main()
