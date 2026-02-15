#!/usr/bin/env bash
# Build MinkowskiEngine v0.5.4 with CUDA 13.0, GCC 13, and NumPy 2.x patches.
#
# Usage:
#   bash build.sh              # build + editable install + verify
#   bash build.sh --build-only # build without installing
#
# Environment variables (auto-detected if unset):
#   CUDA_HOME           — CUDA toolkit root (e.g. /usr/local/cuda-13.0)
#   TORCH_CUDA_ARCH_LIST — GPU compute capabilities (e.g. "8.6" or "8.6;8.9")
#
# Prerequisites:
#   - Active Python venv with PyTorch + CUDA installed
#   - System packages: libopenblas-dev, gcc/g++ matching CUDA requirements

set -euo pipefail

ME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: No active Python venv. Activate one first."
    exit 1
fi

python -c "import torch" 2>/dev/null || {
    echo "ERROR: PyTorch not installed in current venv."
    exit 1
}

# ---------------------------------------------------------------------------
# Auto-detect CUDA_HOME
# ---------------------------------------------------------------------------

if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc &>/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [[ -d /usr/local/cuda ]]; then
        CUDA_HOME=/usr/local/cuda
    else
        echo "ERROR: Cannot find CUDA. Set CUDA_HOME explicitly."
        exit 1
    fi
fi
export CUDA_HOME

NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "CUDA_HOME:  $CUDA_HOME  (nvcc $NVCC_VERSION)"

# ---------------------------------------------------------------------------
# Auto-detect TORCH_CUDA_ARCH_LIST from current GPU
# ---------------------------------------------------------------------------

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST=$(python -c "
import torch
caps = set()
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    caps.add(f'{major}.{minor}')
print(';'.join(sorted(caps)))
" 2>/dev/null) || {
        echo "ERROR: Cannot detect GPU arch. Set TORCH_CUDA_ARCH_LIST explicitly."
        exit 1
    }
fi
export TORCH_CUDA_ARCH_LIST
echo "ARCH_LIST:  $TORCH_CUDA_ARCH_LIST"

# ---------------------------------------------------------------------------
# CCCL include path (needed for CUDA >= 13.0 where thrust headers moved)
# ---------------------------------------------------------------------------

CCCL_INCLUDE="$CUDA_HOME/targets/x86_64-linux/include/cccl"
LOCAL_INCLUDE="$ME_DIR/local_include"

CPLUS_INCLUDE_PATH="${LOCAL_INCLUDE}"
if [[ -d "$CCCL_INCLUDE" ]]; then
    CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${CCCL_INCLUDE}"
fi
export CPLUS_INCLUDE_PATH
echo "INCLUDE:    $CPLUS_INCLUDE_PATH"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

echo ""
echo "Building MinkowskiEngine..."
cd "$ME_DIR"
rm -rf build

python setup.py build_ext --blas=openblas

if $BUILD_ONLY; then
    echo ""
    echo "Build complete (--build-only, skipping install)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Install (editable)
# ---------------------------------------------------------------------------

echo ""
echo "Installing (editable)..."
pip install -e . --no-build-isolation

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

echo ""
echo "Verifying installation..."
python -c "
import MinkowskiEngine as ME
print(f'MinkowskiEngine {ME.__version__} installed successfully')

import torch
if torch.cuda.is_available():
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32).cuda()
    feats = torch.randn(2, 4).cuda()
    x = ME.SparseTensor(features=feats, coordinates=coords)
    conv = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).cuda()
    y = conv(x)
    print(f'Smoke test: {x.F.shape} -> {y.F.shape}  OK')
else:
    print('CUDA not available, skipping smoke test')
"

echo ""
echo "Done."
