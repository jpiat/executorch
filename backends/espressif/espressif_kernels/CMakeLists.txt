# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ESP-IDF component providing the `espressif::` operator kernels on top of
# esp-nn. Add this directory to EXTRA_COMPONENT_DIRS and require
# `espressif_kernels` from the application component; esp-nn is fetched by the
# IDF component manager (see idf_component.yml).
#
# The kernels register themselves with the ExecuTorch runtime from static
# initializers, so the component is linked as a whole archive.

idf_component_register(
  SRCS
  "op_quantize_per_tensor.cpp"
  "op_quantized_conv2d.cpp"
  "op_quantized_elementwise.cpp"
  "op_quantized_linear.cpp"
  "op_quantized_pool2d.cpp"
  "op_quantized_softmax.cpp"
  INCLUDE_DIRS
  "."
  REQUIRES
  espressif__esp-nn
  WHOLE_ARCHIVE
)

set(ET_DIR_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../.."
    CACHE PATH "Path to ExecuTorch source dir"
)
set(ET_BUILD_DIR_PATH
    "${ET_DIR_PATH}/cmake-out-esp"
    CACHE PATH "Path to ExecuTorch build/install dir for ESP target"
)

# Imported targets must be global: ESP-IDF resolves link dependencies from the
# project scope, and the runner component may also look the package up.
if(NOT TARGET executorch)
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
  find_package(
    executorch REQUIRED HINTS "${ET_BUILD_DIR_PATH}/lib/cmake/ExecuTorch"
  )
endif()

target_include_directories(
  ${COMPONENT_LIB} PRIVATE ${ET_DIR_PATH}/..
                           ${ET_DIR_PATH}/runtime/core/portable_type/c10
)
target_compile_definitions(
  ${COMPONENT_LIB} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)
target_link_libraries(${COMPONENT_LIB} PUBLIC executorch)
