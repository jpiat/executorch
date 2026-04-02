# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ESP-IDF project for the MNIST digit classifier example.
#
# Uses the executor_runner component from the parent espressif directory.
#
# Usage:
#   cd examples/espressif/mnist_digit_classifier/project
#   idf.py set-target esp32s3
#   idf.py build
#   idf.py flash monitor

cmake_minimum_required(VERSION 3.16)

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE PATH "ExecuTorch root")

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../executor_runner"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(executorch_esp_mnist)
