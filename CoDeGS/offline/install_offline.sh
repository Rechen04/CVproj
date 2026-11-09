#!/usr/bin/env bash
# Run ON THE OFFLINE SERVER inside the copied offline_bundle/ folder:  bash install_offline.sh
# Installs an isolated Miniconda (Python 3.10) at $CONDA_DIR, pip-installs all wheels offline,
# and wires data/weights into CoDeGS/. Nothing touches your existing Python/conda.
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3_codegs}

# 1) isolated python 3.10
if [ ! -x "$CONDA_DIR/bin/python" ]; then
  echo "[1/4] installing Miniconda (py3.10) -> $CONDA_DIR"
  bash "$B/Miniconda3-py310-Linux-x86_64.sh" -b -p "$CONDA_DIR"
fi
PY="$CONDA_DIR/bin/python"
"$PY" --version

# 2) wheels (fully offline)
echo "[2/4] pip install --no-index"
PIP="$PY -m pip install --no-index --find-links $B/wheels"
$PIP "$B"/wheels/torch-2.4.1*.whl "$B"/wheels/torchvision-0.19.1*.whl
$PIP "$B"/wheels/gsplat-1.5.3*.whl
$PIP numpy==1.26.4 scipy==1.13.1 imageio==2.35.1 imageio-ffmpeg==0.5.1 opencv-python-headless==4.10.0.84 \
     matplotlib==3.9.2 tqdm lpips==0.1.4 transformers==4.46.3 jaxtyping rich ninja

# 3) weights + data
echo "[3/4] weights & data"
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
cp -n "$B"/torch_hub_checkpoints/*.pth "$HOME/.cache/torch/hub/checkpoints/" 2>/dev/null || true
mkdir -p "$B/CoDeGS/weights" "$B/CoDeGS/data"
cp -rn "$B"/weights/* "$B/CoDeGS/weights/" 2>/dev/null || true
if [ -f "$B/nerf_llff_data.zip" ] && [ ! -d "$B/CoDeGS/data/nerf_llff_data" ]; then
  echo "  unzipping LLFF ..."
  (cd "$B/CoDeGS/data" && "$PY" -m zipfile -e "$B/nerf_llff_data.zip" .)
fi
ls "$B/CoDeGS/data/nerf_llff_data" 2>/dev/null || echo "!! LLFF missing: unzip nerf_llff_data.zip into CoDeGS/data/"

# 4) verify
echo "[4/4] verify"
cd "$B/CoDeGS"
"$PY" - <<'EOF'
import torch, gsplat, transformers, lpips
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("gsplat", gsplat.__version__, "| transformers", transformers.__version__)
from gsplat import rasterization
m = torch.randn(100, 3, device="cuda") + torch.tensor([0, 0, 3.0], device="cuda")
r, a, _ = rasterization(m, torch.randn(100, 4, device="cuda"), torch.rand(100, 3, device="cuda") * .05,
                        torch.rand(100, device="cuda"), torch.rand(100, 3, device="cuda"),
                        torch.eye(4, device="cuda")[None], torch.tensor([[[100., 0, 50], [0, 100, 50], [0, 0, 1]]], device="cuda"), 100, 100)
print("gsplat CUDA rasterization OK", tuple(r.shape))
from transformers import AutoModelForDepthEstimation
AutoModelForDepthEstimation.from_pretrained("weights/Depth-Anything-V2-Large-hf", local_files_only=True)
print("Depth-Anything-V2 Large loads OK")
lpips.LPIPS(net="vgg", verbose=False); print("LPIPS(vgg) OK")
EOF
cat <<EOF

All set. Every session:
    source $CONDA_DIR/bin/activate
    cd $B/CoDeGS
    bash scripts/run_smoke.sh
EOF
