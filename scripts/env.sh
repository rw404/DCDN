#!/usr/bin/env bash
# Source this file before every local DCDN environment or experiment command.

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "scripts/env.sh requires bash" >&2
    return 2 2>/dev/null || exit 2
fi

_dcdn_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export DCDN_PROJECT_ROOT="$(cd -- "${_dcdn_script_dir}/.." && pwd -P)"
unset _dcdn_script_dir

# The repository and uv.lock stay persistent. Environments and caches do not.
export UV_PROJECT_ENVIRONMENT="/tmp/dcdn-venv-py312"
export UV_CACHE_DIR="/home/codex/.cache/dcdn/uv"
export XDG_CACHE_HOME="/home/codex/.cache/dcdn/xdg"
export TORCH_HOME="/home/codex/.cache/dcdn/torch"
export HF_HOME="/home/codex/.cache/dcdn/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/codex/.cache/dcdn/huggingface/hub"
export DCDN_DATA_CACHE="/home/codex/.cache/dcdn-public"

# Reproducibility and isolation from user-site packages.
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Disable third-party telemetry and network-backed experiment tracking.
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export UV_KEYRING_PROVIDER=disabled
export UV_PYTHON_DOWNLOADS=never
export UV_NO_PROGRESS=1

# Never let subprocesses implicitly reuse service credentials.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY

# Avoid accidentally inheriting an unrelated active environment.
if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${UV_PROJECT_ENVIRONMENT}" ]]; then
    echo "warning: deactivate the unrelated environment: ${VIRTUAL_ENV}" >&2
fi
