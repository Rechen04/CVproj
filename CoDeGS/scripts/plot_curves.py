"""Test-PSNR vs iteration curves (mean over scenes) for the 3-view runs -> PNG for slides."""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = {"3dgs": "3DGS", "depth_sfm": "3DGS + depth prior", "dinit_conf": "DInit + CW-depth", "full": "CoDe-GS (ours)",
         "3dgs_random": "3DGS (random init)", "dinit": "DInit", "dinit_depth": "DInit + depth", "full_noconf": "full w/o conf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--presets", nargs="*", default=["3dgs", "depth_sfm", "dinit_conf", "full"])
    ap.add_argument("--out", default="assets/figures/curves.png")
    a = ap.parse_args()
    curves = defaultdict(list)
    for f in glob.glob(os.path.join(a.results, "*", "*", "metrics.json")):
        d = json.load(open(f))
        if d["n_views"] == a.n_views and d["preset"] in a.presets and not d.get("tag"):
            curves[d["preset"]].append(d["curve"])
    if not curves:
        raise SystemExit("no curves")
    plt.figure(figsize=(6, 4), dpi=150)
    for p in a.presets:
        if p not in curves:
            continue
        steps = [c["step"] for c in curves[p][0]]
        ps = np.array([[c["psnr"] for c in cv] for cv in curves[p] if len(cv) == len(steps)])
        plt.plot(steps, ps.mean(0), marker="o", label=f"{LABEL.get(p, p)} (n={len(ps)})", lw=2.5 if p == "full" else 1.5)
    plt.xlabel("iteration")
    plt.ylabel("test PSNR (dB)")
    plt.title(f"LLFF, {a.n_views} training views")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
