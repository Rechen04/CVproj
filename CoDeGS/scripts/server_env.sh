#!/usr/bin/env bash
# Source this file on the training server before running project commands.
export CODEGS_SETUP=${CODEGS_SETUP:-/chenpx/codegs_setup}
export PATH="$CODEGS_SETUP/miniconda/bin:$PATH"
export TMPDIR="$CODEGS_SETUP/tmp"
export TORCH_HOME="$CODEGS_SETUP/torch"
export HF_HOME="$CODEGS_SETUP/huggingface"
export XDG_CACHE_HOME="$CODEGS_SETUP/cache"
export MPLCONFIGDIR="$CODEGS_SETUP/cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DM=${DM:-weights/Depth-Anything-V2-Large-hf-fp16}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p "$TMPDIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR"
