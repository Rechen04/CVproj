#!/usr/bin/env bash
# Day-2: small hyper-parameter sweep for OUR method on 3 "dev" scenes only (fern, horns, room).
# Choose the best setting here, then freeze it for the full 8-scene table (state this in the report).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/nerf_llff_data}
DM=${DM:-weights/Depth-Anything-V2-Large-hf}
JOBS=${JOBS:-4}          # parallel runs on one A100 (sparse-view 3DGS uses little memory)
OUT=${OUT:-results_sweep}
SCENES=${SCENES:-"fern horns room"}

python scripts/precompute_depth.py --data_root "$DATA" --scenes $SCENES --depth_model "$DM"

cmds=()
for s in $SCENES; do
  cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset 3dgs --no_ply")
  for wd in 0.05 0.1 0.2; do
    cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset dinit_conf --w_depth_global $wd --w_depth_local $wd --tag wd$wd --no_ply --depth_model $DM")
  done
  for tau in 0.05 0.2; do
    cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset dinit_conf --conf_tau $tau --tag tau$tau --no_ply --depth_model $DM")
  done
  for wp in 0.02 0.05 0.1; do
    cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset full --w_pseudo $wp --tag wp$wp --no_ply --depth_model $DM")
  done
  cmds+=("python train.py --data_dir $DATA/$s --out_dir $OUT --preset full --steps 10000 --tag s10k --no_ply --depth_model $DM")
done
mkdir -p "$OUT/logs"
status=0
if [ "${#cmds[@]}" -gt 0 ]; then
  printf '%s\n' "${cmds[@]}" | nl -nln | xargs -P "$JOBS" -L 1 bash -c 'echo "[start] $*"; "$@" > '"$OUT"'/logs/job_$0.log 2>&1 || { echo "[FAIL] $* (see logs/job_$0.log)"; exit 1; }' || status=1
fi
python scripts/summarize.py --results "$OUT" --out "$OUT/SUMMARY.md"
exit "$status"
