#!/usr/bin/env bash
# Run this ON THE OFFLINE A100 SERVER first. Paste the output back to Claude/yourself to pick the CUDA variant.
echo "== OS / glibc";       uname -a; (ldd --version 2>/dev/null | head -1) || true; cat /etc/os-release 2>/dev/null | head -2
echo "== GPU / driver";     nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not found"
nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" || true
echo "== nvcc (not required)"; (nvcc --version 2>/dev/null | tail -1) || echo "no nvcc (fine: prebuilt wheels)"
echo "== python";           for p in python3 python3.10 python; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1) at $(command -v $p)"; done
echo "== conda";            (conda --version 2>/dev/null) || echo "no conda"
echo "== existing torch";   python3 -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>/dev/null || echo "no torch"
echo "== disk (need ~15 GB)"; df -h "$HOME" | tail -1
echo "== ffmpeg";           (ffmpeg -version 2>/dev/null | head -1) || echo "no system ffmpeg (fine: imageio-ffmpeg bundles one)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "$DRV" ]; then
  if   [ "$DRV" -ge 550 ]; then echo ">>> recommended CUDA variant: cu124"
  elif [ "$DRV" -ge 530 ]; then echo ">>> recommended CUDA variant: cu121"
  else                          echo ">>> recommended CUDA variant: cu118"; fi
fi
