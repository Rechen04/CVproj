#!/usr/bin/env bash
# Day-1 sanity check on the A100: environment + 2 short runs on fern (~3 min total).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data/nerf_llff_data}
DM=${DM:-weights/Depth-Anything-V2-Large-hf}

python -c "import torch,gsplat;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.get_device_name(0),'gsplat',gsplat.__version__)"
python -c "
import torch; from gsplat import rasterization
m=torch.randn(1000,3,device='cuda')+torch.tensor([0,0,3.],device='cuda')
q=torch.randn(1000,4,device='cuda'); s=torch.rand(1000,3,device='cuda')*0.05; o=torch.rand(1000,device='cuda'); c=torch.rand(1000,3,device='cuda')
K=torch.tensor([[[300.,0,160],[0,300,120],[0,0,1]]],device='cuda'); V=torch.eye(4,device='cuda')[None]
r,a,info=rasterization(m,q,s,o,c,V,K,320,240,render_mode='RGB+ED'); print('gsplat CUDA OK', r.shape)"

python scripts/precompute_depth.py --data_root "$DATA" --scenes fern --depth_model "$DM"
python train.py --data_dir "$DATA/fern" --out_dir results_smoke --preset 3dgs --steps 1000 --eval_every 500
python train.py --data_dir "$DATA/fern" --out_dir results_smoke --preset full --steps 1000 --eval_every 500 --pseudo_start 500 --depth_model "$DM"
python scripts/summarize.py --results results_smoke --out results_smoke/SUMMARY.md
