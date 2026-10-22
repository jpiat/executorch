/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "espressif_ops_common.h"

namespace espressif {
namespace native {

namespace {

// Pre-shift applied to both add operands, as in TFLite Micro.
constexpr int32_t kAddLeftShift = 20;

bool check_elementwise_args(
    KernelRuntimeContext& context,
    const char* op,
    const Tensor& self,
    const Tensor& other,
    const Tensor& out) {
  // esp-nn kernels walk all three buffers linearly, so the operands must
  // match the output exactly, including dim order.
  if (!is_int8(self) || !is_int8(other) || !is_int8(out) ||
      self.numel() != other.numel() || self.numel() != out.numel() ||
      !executorch::runtime::tensors_have_same_dim_order(self, other, out)) {
    ET_LOG(Error, "%s: expected same-shaped int8 tensors", op);
    context.fail(Error::InvalidArgument);
    return false;
  }
  return true;
}

} // namespace

Tensor& quantized_add_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& other,
    int64_t self_offset,
    int64_t self_multiplier,
    int64_t self_shift,
    int64_t other_offset,
    int64_t other_multiplier,
    int64_t other_shift,
    int64_t output_offset,
    int64_t output_multiplier,
    int64_t output_shift,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  if (!check_elementwise_args(
          context, "espressif::quantized_add", self, other, out)) {
    return out;
  }
  esp_nn_add_elementwise_s8(
      self.const_data_ptr<int8_t>(),
      other.const_data_ptr<int8_t>(),
      static_cast<int32_t>(self_offset),
      static_cast<int32_t>(other_offset),
      static_cast<int32_t>(self_multiplier),
      static_cast<int32_t>(other_multiplier),
      static_cast<int32_t>(self_shift),
      static_cast<int32_t>(other_shift),
      kAddLeftShift,
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(output_offset),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift),
      static_cast<int32_t>(activation_min),
      static_cast<int32_t>(activation_max),
      static_cast<int32_t>(out.numel()));
  return out;
}

Tensor& quantized_mul_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    const Tensor& other,
    int64_t self_offset,
    int64_t other_offset,
    int64_t output_offset,
    int64_t output_multiplier,
    int64_t output_shift,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  if (!check_elementwise_args(
          context, "espressif::quantized_mul", self, other, out)) {
    return out;
  }
  esp_nn_mul_elementwise_s8(
      self.const_data_ptr<int8_t>(),
      other.const_data_ptr<int8_t>(),
      static_cast<int32_t>(self_offset),
      static_cast<int32_t>(other_offset),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(output_offset),
      static_cast<int32_t>(output_multiplier),
      static_cast<int32_t>(output_shift),
      static_cast<int32_t>(activation_min),
      static_cast<int32_t>(activation_max),
      static_cast<int32_t>(out.numel()));
  return out;
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantized_add.out",
    espressif::native::quantized_add_out);
EXECUTORCH_LIBRARY(
    espressif,
    "quantized_mul.out",
    espressif::native::quantized_mul_out);
