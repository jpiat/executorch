#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-emulator}"
export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif}"
mkdir -p "${build_dir}"
build_dir=$(cd "${build_dir}" && pwd)
cd "${et_root}"

collect_artifacts() {
    if [[ -n "${RUNNER_ARTIFACT_DIR:-}" ]]; then
        mkdir -p "${RUNNER_ARTIFACT_DIR}/esp32s3"
        for file in "${build_dir}"/*.log "${build_dir}"/*.bpte \
            "${build_dir}/firmware/merged-binary.bin" \
            "${build_dir}/firmware/executorch_esp_runner.elf"; do
            if [[ -f "${file}" ]]; then
                cp "${file}" "${RUNNER_ARTIFACT_DIR}/esp32s3/"
            fi
        done
    fi
}
trap collect_artifacts EXIT

bash examples/espressif/setup.sh --minimal 2>&1 | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

python examples/espressif/export_smoke_model.py \
    --output "${build_dir}/smoke.bpte" 2>&1 | tee "${build_dir}/export.log"

cmake --preset esp-baremetal -G Ninja -B "${build_dir}/runtime" \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
    -DEXECUTORCH_SELECT_OPS_LIST=aten::add.out,aten::mul.out \
    2>&1 | tee "${build_dir}/runtime-configure.log"
cmake --build "${build_dir}/runtime" \
    2>&1 | tee "${build_dir}/runtime-build.log"
cmake --install "${build_dir}/runtime" \
    2>&1 | tee "${build_dir}/runtime-install.log"

idf.py -C examples/espressif/project -B "${build_dir}/firmware" \
    -DIDF_TARGET=esp32s3 \
    -DSDKCONFIG="${build_dir}/sdkconfig" \
    -DET_BUILD_DIR_PATH="${build_dir}/runtime" \
    -DET_PTE_FILE_PATH="${build_dir}/smoke.bpte" \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=3 \
    -DET_ATOL=0.000001 -DET_RTOL=0.000001 reconfigure \
    2>&1 | tee "${build_dir}/firmware-configure.log"
cmake --build "${build_dir}/firmware" \
    2>&1 | tee "${build_dir}/firmware-build.log"
idf.py -C examples/espressif/project -B "${build_dir}/firmware" merge-bin \
    2>&1 | tee "${build_dir}/firmware-merge.log"

timeout 130s esp-emu --chip esp32s3 \
    --firmware "${build_dir}/firmware/merged-binary.bin" \
    --elf "${build_dir}/firmware/executorch_esp_runner.elf" \
    --net user --psram-size 8M --timeout 120s --exit-on 'Program complete.' \
    2>&1 | tee "${build_dir}/emulator.log"
python examples/espressif/test_emulator.py "${build_dir}/emulator.log"

# Quantized model on the esp-nn kernels of the Espressif backend.
#
# The emulator does not implement the ESP32-S3 SIMD instructions, so this run
# uses esp-nn's portable C kernels. The bundled reference outputs come from the
# host reference operators, which esp-nn must reproduce exactly.
pip install --no-dependencies -r backends/arm/requirements-arm-tosa.txt \
    2>&1 | tee "${build_dir}/quantized-pip.log"
python -m pytest -q backends/espressif/test \
    2>&1 | tee "${build_dir}/quantized-pytest.log"

python examples/espressif/export_quantized_model.py --model cnn \
    --output "${build_dir}/quantized.bpte" 2>&1 | tee "${build_dir}/quantized-export.log"

# The registry is sized for the selected portable kernels unless told
# otherwise; leave room for the espressif kernels.
cmake --preset esp-baremetal -G Ninja -B "${build_dir}/quantized-runtime" \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF \
    -DEXECUTORCH_SELECT_OPS_LIST="$(tr -d '\n' < "${build_dir}/quantized.ops")" \
    -DMAX_KERNEL_NUM=64 \
    2>&1 | tee "${build_dir}/quantized-runtime-configure.log"
cmake --build "${build_dir}/quantized-runtime" \
    2>&1 | tee "${build_dir}/quantized-runtime-build.log"
cmake --install "${build_dir}/quantized-runtime" \
    2>&1 | tee "${build_dir}/quantized-runtime-install.log"

idf.py -C examples/espressif/project -B "${build_dir}/quantized-firmware" \
    -DIDF_TARGET=esp32s3 \
    -DSDKCONFIG="${build_dir}/quantized-sdkconfig" \
    -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.esp_nn_ansi" \
    -DET_BUILD_DIR_PATH="${build_dir}/quantized-runtime" \
    -DET_PTE_FILE_PATH="${build_dir}/quantized.bpte" \
    -DET_ESPRESSIF_KERNELS=ON \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=3 \
    -DET_ATOL=0.000001 -DET_RTOL=0 reconfigure \
    2>&1 | tee "${build_dir}/quantized-firmware-configure.log"
cmake --build "${build_dir}/quantized-firmware" \
    2>&1 | tee "${build_dir}/quantized-firmware-build.log"
idf.py -C examples/espressif/project -B "${build_dir}/quantized-firmware" merge-bin \
    2>&1 | tee "${build_dir}/quantized-firmware-merge.log"

timeout 130s esp-emu --chip esp32s3 \
    --firmware "${build_dir}/quantized-firmware/merged-binary.bin" \
    --elf "${build_dir}/quantized-firmware/executorch_esp_runner.elf" \
    --net user --psram-size 8M --timeout 120s --exit-on 'Program complete.' \
    2>&1 | tee "${build_dir}/quantized-emulator.log"
python examples/espressif/test_emulator.py "${build_dir}/quantized-emulator.log"
