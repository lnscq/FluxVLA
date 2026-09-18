#!/usr/bin/env bash
# Isolated overlay: no installation or patching into the two existing envs.
set -euo pipefail
RL_ROOT=${RL_ROOT:-/mnt/data/cpfs/users/danny/fluxvla_robotwin_rl}
RL_ENV="$RL_ROOT/envs/flux-robotwin"
export UV_CACHE_DIR="$RL_ROOT/cache/uv"
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export PIP_CACHE_DIR="$RL_ROOT/cache/pip"
export TMPDIR="$RL_ROOT/cache/tmp"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
mkdir -p "$TMPDIR" "$RL_ROOT/envs"
if [[ ! -x "$RL_ENV/bin/python" ]]; then
    /root/miniconda3/envs/fluxvla/bin/python -m venv --system-site-packages "$RL_ENV"
fi
# RLinf dependencies installed in the earlier bridge overlay. Addsitedir also
# processes editable-install .pth files; all NEW packages stay in RL_ENV.
"$RL_ENV/bin/python" -c 'import pathlib,sys; p=pathlib.Path(sys.prefix)/"lib/python3.10/site-packages/rlinf_overlay.pth"; p.write_text("import site; site.addsitedir(\"/root/.venvs/fluxvla-rlinf/lib/python3.10/site-packages\")\n")'
uv pip install --python "$RL_ENV/bin/python" \
    'sapien==3.0.0b1' 'mplib==0.2.1' 'gymnasium==0.29.1' \
    'open3d==0.18.0' 'warp-lang==1.11.1' 'numpy==1.26.4' 'zarr<3' \
    'opencv-python==4.10.0.84' 'av' 'transforms3d' 'toppra' 'openai' 'ninja' \
    'trimesh==4.4.3' 'yourdfpy'
uv pip install --python "$RL_ENV/bin/python" --no-deps --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
uv pip install --python "$RL_ENV/bin/python" --no-deps --no-build-isolation \
    'git+https://github.com/NVlabs/curobo.git@v0.7.7'
# Reproduce RoboTwin script/_install.sh's executable patches, scoped only to
# this new environment. The upstream comment suggests a double-dot SRDF name,
# but its actual sed command does not change that suffix; retain foo.srdf.
RL_SITE="$RL_ENV/lib/python3.10/site-packages"
sed -i -e 's/open(urdf_file, "r")/open(urdf_file, "r", encoding="utf-8")/' \
    -e 's/open(srdf_file, "r")/open(srdf_file, "r", encoding="utf-8")/' \
    "$RL_SITE/sapien/wrapper/urdf_loader.py"
sed -i 's/if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:/if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:/' \
    "$RL_SITE/mplib/planner.py"
"$RL_ENV/bin/python" -m pip freeze > "$RL_ROOT/logs/environment-freeze.txt"
"$RL_ENV/bin/python" -c 'import torch,transformers,numpy,sapien,mplib,rlinf; assert torch.__version__.startswith("2.8."); assert transformers.__version__=="5.3.0"; assert numpy.__version__=="1.26.4"; print("ENV_IMPORTS_READY",torch.__version__,transformers.__version__,numpy.__version__)'
