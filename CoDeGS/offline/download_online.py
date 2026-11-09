"""Cross-platform offline bundle builder. Run on ANY machine with internet (Windows / macOS / Linux,
any Python >= 3.8). Produces ./offline_bundle/ to copy to the A100 server.

    python offline/download_online.py --cuda cu124            # (default)  driver >= 550
    python offline/download_online.py --cuda cu121            #            driver >= 530
    python offline/download_online.py --cuda cu118            #            older drivers
    python offline/download_online.py --cn                    # use Chinese mirrors (Tsinghua PyPI, hf-mirror)
    python offline/download_online.py --skip llff miniconda   # skip parts you already have

Contents (~6-7 GB): Miniconda (py3.10), all linux cp310 wheels (torch 2.4.1 + CUDA libs, gsplat 1.5.3
prebuilt), Depth-Anything-V2 Large+Small (HF format), torchvision VGG/AlexNet weights for LPIPS,
LLFF dataset (nerf_llff_data.zip), and this code repository.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PY_REQS = [
    "numpy==1.26.4", "scipy==1.13.1", "imageio==2.35.1", "imageio-ffmpeg==0.5.1",
    "opencv-python-headless==4.10.0.84", "matplotlib==3.9.2", "tqdm", "lpips==0.1.4",
    "transformers==4.46.3", "jaxtyping", "rich", "ninja", "gdown",
]
PLATFORMS = ["linux_x86_64", "manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64",
             "manylinux_2_17_x86_64", "manylinux_2_27_x86_64", "manylinux_2_28_x86_64"]
GSPLAT = "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24{cu}-cp310-cp310-linux_x86_64.whl"
MINICONDA = {"global": "https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh",
             "cn": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh"}
TV_WEIGHTS = ["https://download.pytorch.org/models/vgg16-397923af.pth",
              "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"]
HF_MODELS = ["depth-anything/Depth-Anything-V2-Large-hf", "depth-anything/Depth-Anything-V2-Small-hf"]
HF_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
LLFF_GDRIVE_ID = "16VnMcF1KJYxN9QId6TClMsZRahHNMW5g"  # nerf_llff_data.zip (official NeRF Google Drive)


def fetch(url, dst):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers, method="HEAD"), timeout=60) as r:
                expected = int(r.headers["Content-Length"]) if r.headers.get("Content-Length") else None
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        if expected is not None and os.path.getsize(dst) == expected:
            print("  have", os.path.basename(dst))
            return
        os.replace(dst, dst + ".part")
    print("  get ", url)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for attempt in range(20):
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected is not None and offset == expected:
            os.replace(tmp, dst)
            return
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            req = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                append = r.status == 206 and offset > 0
                if append and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise IOError("download resumed at an unexpected offset")
                if not append:
                    offset = 0
                length = r.headers.get("Content-Length")
                if length:
                    expected = offset + int(length)
                with open(tmp, "ab" if append else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if expected is None or os.path.getsize(tmp) != expected:
                raise IOError(f"incomplete download: {os.path.getsize(tmp)}/{expected} bytes")
            os.replace(tmp, dst)
            return
        except (OSError, ValueError) as e:
            if attempt == 19:
                raise
            print(f"  retry {os.path.basename(dst)}: {e}", flush=True)
            time.sleep(min(attempt + 1, 5))


def pip_download(args, dest, index=None, extra=None):
    cmd = [sys.executable, "-m", "pip", "download", "-d", dest, "--only-binary=:all:",
           "--python-version", "3.10", "--implementation", "cp", "--abi", "cp310"]
    for p in PLATFORMS:
        cmd += ["--platform", p]
    if index:
        cmd += ["--index-url", index]
    if extra:
        cmd += ["--extra-index-url", extra]
    cmd += args
    print("  $", " ".join(cmd[3:8]), "...", " ".join(args)[:150])
    subprocess.check_call(cmd)


def torch_cuda_deps(torch_whl):
    """Read Requires-Dist of the torch wheel and return the Linux-only deps (nvidia-*, triton) that
    `pip download --platform` silently skips because their markers depend on the *local* machine."""
    with zipfile.ZipFile(torch_whl) as z:
        meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]
        txt = z.read(meta).decode()
    deps = []
    for line in txt.splitlines():
        if line.startswith("Requires-Dist:") and "platform_system" in line and "extra ==" not in line:
            spec = line.split(":", 1)[1].split(";")[0].strip()
            spec = re.sub(r"\s*\((.*)\)", r"\1", spec).replace(" ", "")
            deps.append(spec)
    return deps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default="cu124", choices=["cu124", "cu121", "cu118"])
    ap.add_argument("--out", default="offline_bundle")
    ap.add_argument("--cn", action="store_true", help="use Chinese mirrors where possible")
    ap.add_argument("--skip", nargs="*", default=[], choices=["miniconda", "wheels", "models", "lpips", "llff", "repo"])
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    pypi = "https://pypi.tuna.tsinghua.edu.cn/simple" if a.cn else "https://pypi.org/simple"
    torch_idx = f"https://download.pytorch.org/whl/{a.cuda}"
    hf = "https://hf-mirror.com" if a.cn else os.environ.get("HF_ENDPOINT", "https://huggingface.co")

    if "miniconda" not in a.skip:
        print("[1/6] Miniconda (Python 3.10, linux-64)")
        fetch(MINICONDA["cn" if a.cn else "global"], os.path.join(out, "Miniconda3-py310-Linux-x86_64.sh"))

    if "wheels" not in a.skip:
        print(f"[2/6] wheels for linux / cp310 / {a.cuda}")
        wd = os.path.join(out, "wheels")
        os.makedirs(wd, exist_ok=True)
        pip_download(["--no-deps", "torch==2.4.1", "torchvision==0.19.1"], wd, index=torch_idx)
        tw = [f for f in os.listdir(wd) if f.startswith("torch-2.4.1")][0]
        cuda_deps = torch_cuda_deps(os.path.join(wd, tw))
        print("  CUDA runtime deps from torch METADATA:", cuda_deps)
        pip_download(cuda_deps, wd, index=pypi, extra=torch_idx)
        # torch/torchvision pure deps + our requirements
        pip_download(PY_REQS + ["torch==2.4.1", "torchvision==0.19.1", "--find-links", wd,
                                "filelock", "typing-extensions>=4.8.0", "sympy", "networkx", "jinja2", "fsspec",
                                "pillow"], wd, index=pypi, extra=torch_idx)
        fetch(GSPLAT.format(cu=a.cuda), os.path.join(wd, f"gsplat-1.5.3+pt24{a.cuda}-cp310-cp310-linux_x86_64.whl"))
        with open(os.path.join(out, "CUDA_VARIANT.txt"), "w") as f:
            f.write(a.cuda + "\n")

    if "models" not in a.skip:
        print("[3/6] Depth-Anything-V2 (HF transformers format)")
        for m in HF_MODELS:
            for fn in HF_FILES:
                fetch(f"{hf}/{m}/resolve/main/{fn}", os.path.join(out, "weights", m.split("/")[1], fn))

    if "lpips" not in a.skip:
        print("[4/6] torchvision backbones for LPIPS")
        for u in TV_WEIGHTS:
            fetch(u, os.path.join(out, "torch_hub_checkpoints", os.path.basename(u)))

    if "llff" not in a.skip:
        print("[5/6] LLFF dataset (nerf_llff_data.zip, 1.8 GB)")
        dst = os.path.join(out, "nerf_llff_data.zip")
        if not os.path.exists(dst):
            try:
                import gdown  # noqa
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-i", pypi])
            try:
                subprocess.check_call([sys.executable, "-m", "gdown", LLFF_GDRIVE_ID, "-O", dst])
            except Exception as e:  # noqa
                print("  !! Google Drive download failed:", e)
                print("  -> download manually (see offline/README_OFFLINE.md) and put it at", dst)

    if "repo" not in a.skip:
        print("[6/6] code repository")
        shutil.copytree(REPO, os.path.join(out, "CoDeGS"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("offline_bundle", "results*", "cache", "__pycache__", "*.pyc", "assets"))
        shutil.copy(os.path.join(HERE, "install_offline.sh"), os.path.join(out, "install_offline.sh"))
    print("\nDone ->", out, "\nCopy the whole folder to the server, then:  bash install_offline.sh")


if __name__ == "__main__":
    main()
