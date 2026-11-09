#!/usr/bin/env bash
# Day-5: presentation assets — comparison videos + qualitative grids for every scene (3 views).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results}
SCENES=${SCENES:-"fern flower fortress horns leaves orchids room trex"}
mkdir -p assets/videos assets/figures
for s in $SCENES; do
  A=$OUT/$s/3dgs_3v; B=$OUT/$s/depth_sfm_3v; C=$OUT/$s/full_3v
  [ -f "$A/ckpt.pt" ] && [ -f "$C/ckpt.pt" ] || { echo "skip $s"; continue; }
  python render_video.py --ckpts $A/ckpt.pt $C/ckpt.pt --labels "3DGS (3 views)" "CoDe-GS (ours)" \
         --out assets/videos/${s}_3dgs_vs_ours.mp4
  if [ -f "$B/ckpt.pt" ]; then
    python render_video.py --ckpts $A/ckpt.pt $B/ckpt.pt $C/ckpt.pt --labels "3DGS" "+Depth prior" "Ours" \
           --out assets/videos/${s}_3way.mp4 --upscale 1
    python scripts/make_figure.py --scene_dir $OUT/$s --runs 3dgs_3v depth_sfm_3v full_3v \
           --labels "3DGS" "+Depth prior" "Ours" --out assets/figures/${s}_grid.png
  else
    python scripts/make_figure.py --scene_dir $OUT/$s --runs 3dgs_3v full_3v --labels "3DGS" "Ours" \
           --out assets/figures/${s}_grid.png
  fi
  # depth prior / confidence panels for the method slide
  mkdir -p assets/prior/$s && cp $C/prior/*.png assets/prior/$s/ 2>/dev/null || true
done
python scripts/plot_curves.py --results $OUT --out assets/figures/curves.png || true
echo "assets in ./assets  (point_cloud.ply files can be dragged into https://superspl.at/editor for a live 3D demo)"
