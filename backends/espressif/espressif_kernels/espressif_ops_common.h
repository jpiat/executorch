/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

#include "esp_nn.h"

namespace espressif {
namespace native {

using executorch::aten::IntArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;

// esp-nn takes spatial sizes and channel counts as uint16_t.
constexpr int64_t kMaxDim = UINT16_MAX;

// Fail the kernel with InvalidArgument and log the condition when it is false.
#define ESP_NN_CHECK(context, cond, out, ...) \
  do {                                        \
    if (!(cond)) {                            \
      ET_LOG(Error, __VA_ARGS__);             \
      (context).fail(Error::InvalidArgument); \
      return out;                             \
    }                                         \
  } while (false)

inline bool is_int8(const Tensor& t) {
  return t.scalar_type() == ScalarType::Char;
}

// Activations of spatial ops are logically NCHW with NHWC bytes. Tensors with
// a single channel or a 1x1 plane are laid out the same either way.
inline bool is_nhwc(const Tensor& t) {
  if (t.dim() != 4) {
    return false;
  }
  if (t.size(1) == 1 || (t.size(2) == 1 && t.size(3) == 1)) {
    return true;
  }
  return executorch::runtime::is_channels_last_dim_order(
      t.dim_order().data(), t.dim_order().size());
}

inline bool is_contiguous(const Tensor& t) {
  return executorch::runtime::is_contiguous_dim_order(
      t.dim_order().data(), t.dim_order().size());
}

inline bool fits_uint16(int64_t value) {
  return value >= 0 && value <= kMaxDim;
}

// Allocate an esp-nn scratch buffer from the method's temp allocator. esp-nn
// keeps the pointer in a global, so it must be set before every call.
inline bool
allocate_scratch(KernelRuntimeContext& context, int32_t size, void** buffer) {
  *buffer = nullptr;
  if (size <= 0) {
    return true;
  }
  auto result = context.allocate_temp(static_cast<size_t>(size), 16);
  if (!result.ok()) {
    ET_LOG(
        Error,
        "esp-nn needs %ld bytes of scratch; increase the temp allocator pool",
        static_cast<long>(size));
    context.fail(Error::MemoryAllocationFailed);
    return false;
  }
  *buffer = result.get();
  return true;
}

} // namespace native
} // namespace espressif
