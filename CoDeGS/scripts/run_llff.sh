#!/usr/bin/env bash
# Day-3/4: main table + ablation on all 8 LLFF scenes, 3/6/9 views.
#   VIEWS="3" PRESETS="3dgs full" bash scripts/run_llff.sh   # subset
# Pass frozen hyper-parameters from the sweep through EXTRA, e.g. EXTRA="--w_pseudo 0.05".
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/nerf_llff_data}
DM=${DM:-weights/Depth-Anything-V2-Large-hf}
JOBS=${JOBS:-4}
OUT=${OUT:-results}
SCENES=${SCENES:-"fern flower fortress horns leaves orchids room trex"}
VIEWS=${VIEWS:-"3 6 9"}
PRESETS=${PRESETS:-"3dgs_random 3dgs depth_sfm dinit dinit_depth dinit_conf full_noconf full"}
EXTRA=${EXTRA:-""}

python scripts/precompute_depth.py --data_root "$DATA" --scenes $SCENES --depth_model "$DM"

cmds=()
for v in $VIEWS; do
  for p in $PRESETS; do
    # full ablation only at 3 views; 6/9 views: baseline vs ours (+ prior-work style)
    if [ "$v" != "3" ] && [[ ! " 3dgs depth_sfm full " =~ " $p " ]]; then continue; fi
    for s in $SCENES; do
      run_dir="$OUT/$s/${p}_${v}v"
      if [ -f "$run_dir/metrics.json" ] && [ -f "$run_dir/ckpt.pt" ] && [ -f "$run_dir/point_cloud.ply" ]; then continue; fi
      cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset $p --n_views $v --depth_model $DM $EXTRA")
    done
  done
done
echo "${#cmds[@]} runs queued, $JOBS in parallel"
mkdir -p "$OUT/logs"
status=0
if [ "${#cmds[@]}" -gt 0 ]; then
  printf '%s\n' "${cmds[@]}" | nl -nln | xargs -P "$JOBS" -L 1 bash -c 'echo "[start] $*"; "$@" > '"$OUT"'/logs/job_$0.log 2>&1 || { echo "[FAIL] $* (see logs/job_$0.log)"; exit 1; }' || status=1
fi
python scripts/summarize.py --results "$OUT" --out "$OUT/SUMMARY.md"
exit "$status"
