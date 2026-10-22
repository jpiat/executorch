/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>

#include "espressif_ops_common.h"

namespace espressif {
namespace native {

Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  ESP_NN_CHECK(
      context,
      input.scalar_type() == ScalarType::Float && is_int8(out) &&
          dtype == ScalarType::Char && input.numel() == out.numel(),
      out,
      "espressif::quantize_per_tensor: expected float input and int8 output");

  const float inv_scale = 1.0f / static_cast<float>(scale);
  const float qmin = static_cast<float>(quant_min);
  const float qmax = static_cast<float>(quant_max);
  const float zp = static_cast<float>(zero_point);
  const float* in = input.const_data_ptr<float>();
  int8_t* dst = out.mutable_data_ptr<int8_t>();
  for (ssize_t i = 0; i < input.numel(); ++i) {
    // nearbyint rounds half to even, as torch.round does.
    const float q = std::nearbyint(in[i] * inv_scale) + zp;
    dst[i] = static_cast<int8_t>(std::min(std::max(q, qmin), qmax));
  }
  return out;
}

Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  (void)quant_min;
  (void)quant_max;
  ESP_NN_CHECK(
      context,
      is_int8(input) && dtype == ScalarType::Char &&
          out.scalar_type() == ScalarType::Float &&
          input.numel() == out.numel(),
      out,
      "espressif::dequantize_per_tensor: expected int8 input and float output");

  const float s = static_cast<float>(scale);
  const int32_t zp = static_cast<int32_t>(zero_point);
  const int8_t* in = input.const_data_ptr<int8_t>();
  float* dst = out.mutable_data_ptr<float>();
  for (ssize_t i = 0; i < input.numel(); ++i) {
    dst[i] = static_cast<float>(static_cast<int32_t>(in[i]) - zp) * s;
  }
  return out;
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantize_per_tensor.out",
    espressif::native::quantize_per_tensor_out);
EXECUTORCH_LIBRARY(
    espressif,
    "dequantize_per_tensor.out",
    espressif::native::dequantize_per_tensor_out);
