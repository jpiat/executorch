#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build script for the ESP32 MNIST digit classifier example.
#
# Prerequisites:
#   - ESP-IDF v5.1+ installed and sourced (. $IDF_PATH/export.sh)
#   - ExecuTorch cross-compiled for the ESP32 target
#   - Python 3.8+
#
# Usage:
#   ./build.sh [--target esp32|esp32s3] [--pte <model.pte>] [--clean]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ET_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/project"
EXECUTOR_RUNNER_DIR="${SCRIPT_DIR}/../executor_runner"
ET_BUILD_DIR="${ET_ROOT}/cmake-out-esp"
TARGET="esp32s3"
PTE_FILE=""
CLEAN=false
SKIP_ET_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --pte)
            PTE_FILE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --skip-et-build)
            SKIP_ET_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--target esp32|esp32s3] [--pte <model.pte>] [--clean] [--skip-et-build]"
            echo ""
            echo "Options:"
            echo "  --target        ESP32 target chip (default: esp32s3)"
            echo "  --pte           Path to the .pte model file to embed"
            echo "  --clean         Clean build directory before building"
            echo "  --skip-et-build Skip cross-compiling ExecuTorch libraries"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${IDF_PATH:-}" ]; then
    echo "ERROR: IDF_PATH is not set. Please source ESP-IDF:"
    echo "  . \$IDF_PATH/export.sh"
    exit 1
fi

echo "=== ExecuTorch ESP32 MNIST Digit Classifier Build ==="
echo "Target: ${TARGET}"
echo "ExecuTorch root: ${ET_ROOT}"
echo "ESP-IDF: ${IDF_PATH}"

if [ -z "${PTE_FILE}" ]; then
    echo "ERROR: --pte <model.pte> is required."
    echo "  Export a model first: python3 ${SCRIPT_DIR}/export_mlp_mnist.py"
    exit 1
fi

if [ ! -f "${PTE_FILE}" ]; then
    echo "ERROR: PTE file not found: ${PTE_FILE}"
    exit 1
fi

# Resolve to absolute path for use in cmake
PTE_FILE="$(cd "$(dirname "${PTE_FILE}")" && pwd)/$(basename "${PTE_FILE}")"

# Step 1: Cross-compile ExecuTorch with selective build from the model
if [ "${SKIP_ET_BUILD}" = false ]; then
    echo ""
    echo "--- Cross-compiling ExecuTorch for ${TARGET} (selective build from model) ---"
    echo "Using EXECUTORCH_SELECT_OPS_MODEL=${PTE_FILE}"

    IDF_TARGET="${TARGET}"
    TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-${IDF_TARGET}.cmake"

    if [ ! -f "${TOOLCHAIN_FILE}" ]; then
        echo "ERROR: Toolchain file not found: ${TOOLCHAIN_FILE}"
        exit 1
    fi

    cmake --preset esp-baremetal -B "${ET_BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
        -DEXECUTORCH_SELECT_OPS_MODEL="${PTE_FILE}" \
        "${ET_ROOT}"

    cmake --build "${ET_BUILD_DIR}" -j"$(nproc)"
    cmake --build "${ET_BUILD_DIR}" --target install

    echo "ExecuTorch cross-compilation complete: ${ET_BUILD_DIR}"
fi

# Step 2: Convert PTE to header
echo ""
echo "--- Converting PTE to header ---"
HEADER_DIR="${PROJECT_DIR}"
mkdir -p "${HEADER_DIR}"
python3 "${EXECUTOR_RUNNER_DIR}/pte_to_header.py" \
    --pte "${PTE_FILE}" \
    --outdir "${HEADER_DIR}"
echo "Model header generated: ${HEADER_DIR}/model_pte.h"

# Step 3: Build the ESP-IDF project
cd "${PROJECT_DIR}"

if [ "${CLEAN}" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build sdkconfig
fi

echo ""
echo "--- Building ESP-IDF project ---"
echo "Setting target to ${TARGET}..."
idf.py set-target "${TARGET}"

echo "Building..."
idf.py build

echo ""
echo "=== Build complete ==="
echo ""
echo "To flash and monitor:"
echo "  cd ${PROJECT_DIR}"
echo "  idf.py -p /dev/ttyUSB0 flash monitor"
