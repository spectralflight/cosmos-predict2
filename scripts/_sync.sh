#!/usr/bin/env bash

# Synchronize and compile dependencies.
# Used by `just install`.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <CUDA_NAME> <ALL_EXTRAS>"
    exit 1
fi
CUDA_NAME=$1
shift
ALL_EXTRAS=$1
shift

export PATH="$(pwd)/.venv/bin:$PATH"

# Set cuda environment variables
echo "CUDA_NAME: $CUDA_NAME"
CUDA_VERSION="$(echo "$CUDA_NAME" | sed -E 's/cu([0-9]+)([0-9])/\1.\2/')"
echo "CUDA_VERSION: $CUDA_VERSION"
export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
if [ ! -d "$CUDA_HOME" ]; then
    echo "Error: CUDA $CUDA_VERSION not installed. Please install https://developer.nvidia.com/cuda-toolkit-archive" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Install build dependencies
extras="--extra $CUDA_NAME"
uv sync --extra build $extras "$@"

# Set build environment variables
eval $(python -c "
import torch
from packaging.version import Version
print(f'export TORCH_VERSION={Version(torch.__version__).base_version}')
print(f'export _GLIBCXX_USE_CXX11_ABI={1 if torch.compiled_with_cxx11_abi() else 0}')
")
echo "TORCH_VERSION: $TORCH_VERSION"
echo "_GLIBCXX_USE_CXX11_ABI: $_GLIBCXX_USE_CXX11_ABI"
export UV_CACHE_DIR="$(uv cache dir)/torch${TORCH_VERSION//./}_cu${CUDA_VERSION//./}_cxx11abi=${_GLIBCXX_USE_CXX11_ABI}"
# uv requires clang: https://github.com/astral-sh/uv/issues/11707
export CXX=clang
if ! command -v clang &> /dev/null; then
    echo "Error: clang not installed." >&2
    exit 1
fi

export APEX_BUILD_ARGS="--cpp_ext --cuda_ext"

# transformer-engine: https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#pip-installation
export NVTE_FRAMEWORK=pytorch

# export NATTEN_CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;10.3;12.0"
# export NATTEN_VERBOSE=1

# Compile dependencies
for extra in $ALL_EXTRAS; do \
    echo "Compiling $extra. This may take a while..."; \
    extras+=" --extra $extra"; \
    uv sync --extra build $extras "$@"; \
done

# Remove build dependencies
uv sync $extras "$@"
