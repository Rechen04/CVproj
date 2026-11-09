#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh
mkdir -p training

# Transfers can finish independently. Verify complete files before consuming them.
python - <<'PY'
import hashlib, json, os, time
from pathlib import Path
setup = Path(os.environ['CODEGS_SETUP'])
manifest = json.loads((setup / 'verified_manifest.json').read_text())
required = [
    'wheels/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl',
    'torch_hub_checkpoints/vgg16-397923af.pth',
    'torch_hub_checkpoints/alexnet-owt-7be5be79.pth',
] + ['weights/Depth-Anything-V2-Large-hf-fp16/' + name
     for name in ['config.json', 'preprocessor_config.json', 'model.safetensors']]
verified = set()
while len(verified) < len(required):
    waiting = []
    for name in required:
        if name in verified:
            continue
        if name.startswith('weights/'):
            path = Path(name)
        elif name.startswith('torch_hub_checkpoints/'):
            path = setup / 'torch/hub/checkpoints' / Path(name).name
        else:
            path = setup / name
        info = manifest[name]
        size = path.stat().st_size if path.exists() else 0
        if size != info['bytes']:
            waiting.append(f'{path.name}: {size}/{info["bytes"]}')
            continue
        with path.open('rb') as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != info['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}')
        verified.add(name)
        print('Verified upload:', name, flush=True)
    if waiting:
        print('Waiting for uploads:', '; '.join(waiting), flush=True)
        time.sleep(30)
PY

python -m pip install --no-cache-dir --no-index --no-deps \
  "$CODEGS_SETUP/wheels/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl"
python scripts/verify_server.py
exec python scripts/run_managed.py --jobs "${JOBS:-4}"
