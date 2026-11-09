"""Pre-compute Depth-Anything-V2 disparities for every image of every scene (fills the cache that
train.py reads, so parallel training jobs never race on it). Also prints alignment quality.

python scripts/precompute_depth.py --data_root data/nerf_llff_data --depth_model weights/Depth-Anything-V2-Large-hf
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sparsegs import depth_prior as dp  # noqa: E402
from sparsegs.data import load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--depth_model", default="weights/Depth-Anything-V2-Large-hf")
    ap.add_argument("--depth_cache", default="cache/mono_depth")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    scenes = a.scenes or sorted(d for d in os.listdir(a.data_root) if os.path.isdir(os.path.join(a.data_root, d)))
    for s in scenes:
        sc = load_scene(os.path.join(a.data_root, s), factor=a.factor, n_views=3)
        ids = np.arange(len(sc.names))
        mono = dp.compute_mono_disparities(sc, ids, a.depth_model, os.path.join(a.depth_cache, sc.name), device=a.device)
        tr = {int(i): mono[int(i)] for i in sc.train_ids}
        _, info = dp.aligned_depths(sc, tr)
        print(s, f"{len(ids)} images cached;", "3-view align corr:", [round(v["corr"], 3) for v in info.values()])


if __name__ == "__main__":
    main()
