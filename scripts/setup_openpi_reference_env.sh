#!/usr/bin/env bash
set -euo pipefail
RL_ROOT=${RL_ROOT:-/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl}
REF_ENV="$RL_ROOT/envs/openpi-reference"
export UV_CACHE_DIR="$RL_ROOT/cache/uv"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export TMPDIR="$RL_ROOT/cache/tmp"
mkdir -p "$TMPDIR"
if [[ ! -x "$REF_ENV/bin/python" ]]; then
    /root/miniconda3/envs/fluxvla/bin/python -m venv --system-site-packages "$REF_ENV"
fi
# OpenPI's patched Transformers is confined to this reference process. Avoid
# its Torch 2.7 pin / JAX CUDA extra replacing the agreed Torch 2.8 runtime.
uv pip install --python "$REF_ENV/bin/python" --no-deps \
    'rlinf-openpi==0.1.1' 'rlinf-transformer-openpi==4.53.2' 'openpi-client' \
    'tokenizers==0.21.4' 'huggingface-hub==0.35.3' 'flax==0.10.2' \
    'jax==0.5.3' 'jaxlib==0.5.3' 'orbax-checkpoint==0.11.13' \
    'chex==0.1.90' 'jaxtyping==0.2.36' 'beartype==0.19.0' \
    'augmax' 'equinox==0.11.12' 'ml-collections==1.0.0' 'numpydantic==1.7.0' \
    'treescope' 'tqdm-loggable' 'gym-aloha'
uv pip install --python "$REF_ENV/bin/python" \
    'flax==0.10.2' 'jax==0.5.3' 'jaxlib==0.5.3' 'numpy==1.26.4' \
    'orbax-checkpoint==0.11.13' 'chex==0.1.90' 'openpi-client' \
    'augmax' 'ml-collections==1.0.0' 'numpydantic==1.7.0' 'pydantic==2.11.7' 'tqdm-loggable'
"$REF_ENV/bin/python" -m pip freeze > "$RL_ROOT/logs/reference-freeze.txt"
"$REF_ENV/bin/python" -c 'import torch,transformers; from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; print("REFERENCE_IMPORT_READY",torch.__version__,transformers.__version__)'
