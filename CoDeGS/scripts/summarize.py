"""Aggregate metrics.json files into the paper-style tables (Markdown + CSV).

python scripts/summarize.py --results results --out results/SUMMARY.md
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

LLFF = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
ORDER = ["3dgs_random", "3dgs", "depth_sfm", "dinit", "dinit_depth", "dinit_conf", "full_noconf", "full"]
NAMES = {
    "3dgs_random": "3DGS (random init)", "3dgs": "3DGS (original, SfM init)",
    "depth_sfm": "3DGS + mono-depth Pearson (prior-work style)", "dinit": "+ DInit",
    "dinit_depth": "+ DInit + depth prior", "dinit_conf": "+ DInit + CW depth prior",
    "full_noconf": "+ DInit + depth prior + PV", "full": "**CoDe-GS (ours, full)**",
}
KEYS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "n_gaussians", "train_sec", "fps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/SUMMARY.md")
    ap.add_argument("--scenes", nargs="*", default=None, help="restrict to these scenes")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(os.path.join(a.results, "*", "*", "metrics.json")):
        d = json.load(open(f))
        if a.scenes and d["scene"] not in a.scenes:
            continue
        rows.append(d)
    if not rows:
        raise SystemExit("no metrics.json found")
    groups = defaultdict(list)
    for d in rows:
        groups[(d["n_views"], d["preset"], d.get("tag", ""))].append(d)

    lines = ["# Results (mean over scenes; test views = every 8th image)\n"]
    csv_rows = []
    for nv in sorted({k[0] for k in groups}):
        lines.append(f"\n## {nv} training views\n")
        lines.append("| Method | tag | #scenes | PSNR↑ | SSIM↑ | LPIPS(vgg)↓ | LPIPS(alex)↓ | #GS | train s | FPS |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        keys = sorted([k for k in groups if k[0] == nv],
                      key=lambda k: (ORDER.index(k[1]) if k[1] in ORDER else 99, k[2]))
        for k in keys:
            g = groups[k]
            m = {}
            for key in KEYS:
                vals = [d[key] for d in g if d.get(key) is not None]
                m[key] = float(np.mean(vals)) if vals else None
            fmt = lambda v, p: "-" if v is None else f"{v:.{p}f}"  # noqa
            lines.append(f"| {NAMES.get(k[1], k[1])} | {k[2]} | {len(g)} | {fmt(m['psnr'], 2)} | {fmt(m['ssim'], 3)} | "
                         f"{fmt(m['lpips_vgg'], 3)} | {fmt(m['lpips_alex'], 3)} | {fmt(m['n_gaussians'], 0)} | "
                         f"{fmt(m['train_sec'], 0)} | {fmt(m['fps'], 0)} |")
            csv_rows.append(dict(n_views=nv, preset=k[1], tag=k[2], n_scenes=len(g), **m))
        # per-scene PSNR table
        scenes = sorted({d["scene"] for k in keys for d in groups[k]},
                        key=lambda s: LLFF.index(s) if s in LLFF else 99)
        lines.append(f"\n### Per-scene PSNR ({nv} views)\n")
        lines.append("| Method | " + " | ".join(scenes) + " |")
        lines.append("|---|" + "---|" * len(scenes))
        for k in keys:
            by = {d["scene"]: d["psnr"] for d in groups[k]}
            lines.append(f"| {NAMES.get(k[1], k[1])} {k[2]} | " + " | ".join(f"{by[s]:.2f}" if s in by else "-" for s in scenes) + " |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
