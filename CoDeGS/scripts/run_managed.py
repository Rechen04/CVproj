"""Run the documented experiment sequence and persist its progress on the server."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCENES = "fern flower fortress horns leaves orchids room trex".split()
PRESETS = "3dgs_random 3dgs depth_sfm dinit dinit_depth dinit_conf full_noconf full".split()
DEV = ["fern", "horns", "room"]
METRICS = ["psnr", "ssim", "lpips_vgg", "lpips_alex"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate(run, ply=True):
    required = ["metrics.json", "ckpt.pt", "args.json"]
    if ply:
        required.append("point_cloud.ply")
    for name in required:
        if not (run / name).is_file() or (run / name).stat().st_size == 0:
            raise RuntimeError(f"Missing output: {run / name}")
    data = json.loads((run / "metrics.json").read_text())
    if not all(math.isfinite(data.get(k, float("nan"))) for k in METRICS):
        raise RuntimeError(f"Incomplete or non-finite metrics: {run}")
    return data


def choose_setting():
    candidates = []
    for tag in ["wp0.02", "wp0.05", "wp0.1", "s10k"]:
        runs = [ROOT / "results_sweep" / scene / f"full_3v_{tag}" for scene in DEV]
        rows = [validate(run, ply=False) for run in runs]
        args = [json.loads((run / "args.json").read_text()) for run in runs]
        keys = ["steps", "w_depth_global", "w_depth_local", "conf_tau", "w_pseudo"]
        params = {k: args[0][k] for k in keys}
        if any({k: a[k] for k in keys} != params for a in args):
            raise RuntimeError(f"Inconsistent sweep parameters: {tag}")
        candidates.append(dict(tag=tag, parameters=params,
                               means={k: sum(row[k] for row in rows) / len(rows) for k in METRICS}))
    for c in candidates:
        c["rank_sum"] = sum(d["means"]["psnr"] > c["means"]["psnr"] for d in candidates)
        c["rank_sum"] += sum(d["means"]["lpips_vgg"] < c["means"]["lpips_vgg"] for d in candidates)
    chosen = min(candidates, key=lambda c: (c["rank_sum"], -c["means"]["psnr"]))
    result = dict(rule="Lowest sum of mean PSNR rank and mean LPIPS(vgg) rank over fern/horns/room; PSNR breaks ties",
                  development_scenes=DEV, candidates=candidates, chosen=chosen)
    save(ROOT / "training" / "frozen_parameters.json", result)
    return chosen["parameters"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=["smoke", "sweep", "main", "assets"], default="smoke")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.chdir(ROOT)
    progress = ROOT / "training"
    progress.mkdir(exist_ok=True)
    lock = (progress / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An experiment pipeline is already running; see training/status.json")
    env = dict(os.environ, JOBS=str(args.jobs), PYTHONUNBUFFERED="1")
    state = dict(status="running", pid=os.getpid(), jobs=args.jobs, completed_stages=[])

    def update(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        save(progress / "status.json", state)

    def run(stage, script, extra_env=None):
        log = progress / f"{stage}.log"
        update(stage=stage, log=str(log))
        print(f"Starting {stage}: {log}", flush=True)
        with log.open("a") as f:
            subprocess.run(["bash", script], env=dict(env, **(extra_env or {})), stdout=f,
                           stderr=subprocess.STDOUT, check=True)

    try:
        stages = ["smoke", "sweep", "main", "assets"]
        for stage in stages[stages.index(args.start):]:
            if stage == "smoke":
                run(stage, "scripts/run_smoke.sh")
                for preset in ["3dgs", "full"]:
                    validate(ROOT / "results_smoke" / "fern" / f"{preset}_3v")
            elif stage == "sweep":
                run(stage, "scripts/run_sweep.sh", {"SCENES": " ".join(DEV), "OUT": "results_sweep"})
                choose_setting()
            elif stage == "main":
                frozen = json.loads((progress / "frozen_parameters.json").read_text())["chosen"]["parameters"]
                extra = " ".join(f"--{k} {v}" for k, v in frozen.items())
                run(stage, "scripts/run_llff.sh", dict(EXTRA=extra, OUT="results", SCENES=" ".join(SCENES),
                                                       VIEWS="3 6 9", PRESETS=" ".join(PRESETS)))
                for n in [3, 6, 9]:
                    for preset in PRESETS if n == 3 else ["3dgs", "depth_sfm", "full"]:
                        for scene in SCENES:
                            validate(ROOT / "results" / scene / f"{preset}_{n}v")
            else:
                run(stage, "scripts/make_assets.sh", {"OUT": "results", "SCENES": " ".join(SCENES)})
                for scene in SCENES:
                    for path in [ROOT / "assets/videos" / f"{scene}_3dgs_vs_ours.mp4",
                                 ROOT / "assets/figures" / f"{scene}_grid.png"]:
                        if not path.is_file() or path.stat().st_size == 0:
                            raise RuntimeError(f"Missing presentation asset: {path}")
            state["completed_stages"].append(stage)
            update()
        update(status="complete")
    except Exception as exc:
        update(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
