"""Validate the deployed CUDA renderer, depth model, metrics, and eight LLFF scenes."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import gsplat
from gsplat import rasterization
from sparsegs.data import load_scene
from sparsegs.depth_prior import MonoDepth
from sparsegs.utils import Metrics

assert torch.cuda.is_available(), "CUDA is required"
report = dict(torch=torch.__version__, compiled_cuda=torch.version.cuda,
              cudnn=torch.backends.cudnn.version(), gsplat=gsplat.__version__,
              gpu=torch.cuda.get_device_name(0))
means = (torch.randn(100, 3, device="cuda") * .1 + torch.tensor([0., 0., 3.], device="cuda")).requires_grad_()
renders, alpha, _ = rasterization(
    means, torch.randn(100, 4, device="cuda"), torch.full((100, 3), .05, device="cuda"),
    torch.full((100,), .5, device="cuda"), torch.rand(100, 3, device="cuda"),
    torch.eye(4, device="cuda")[None],
    torch.tensor([[[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]]], device="cuda"),
    64, 48, render_mode="RGB+ED")
renders.sum().backward()
assert torch.isfinite(renders).all() and torch.isfinite(means.grad).all()
assert means.grad.abs().sum() > 0
report["renderer_forward_backward"] = "passed"
print(report, flush=True)

model_dir = os.environ.get("DM", "weights/Depth-Anything-V2-Large-hf-fp16")
model = MonoDepth(model_dir)
depth = model(torch.rand(1, 3, 378, 504, device="cuda"))
assert torch.isfinite(depth).all() and depth.std() > 0
report["depth_model"] = model_dir
report["depth_output_shape"] = list(depth.shape)
del model, depth
torch.cuda.empty_cache()

metrics = Metrics("cuda")
assert set(metrics.lpips) == {"vgg", "alex"}, "Both LPIPS backbones are required"
img = torch.rand(64, 64, 3, device="cuda")
metric_values = metrics(img, img)
assert metric_values["psnr"] > 100 and metric_values["lpips_vgg"] < 1e-5
report["metrics"] = metric_values
del metrics
torch.cuda.empty_cache()

report["scenes"] = []
for name in "fern flower fortress horns leaves orchids room trex".split():
    scene = load_scene("data/nerf_llff_data/" + name)
    report["scenes"].append(dict(scene=name, images=len(scene.names), width=scene.W, height=scene.H,
                                  train=scene.train_ids.tolist(), test=scene.test_ids.tolist(),
                                  train_sfm_points=len(scene.train_sfm_points()[0])))
out = ROOT / "training" / "environment_validation.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
print("ALL SERVER CHECKS PASSED", flush=True)
