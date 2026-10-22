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

Tensor& quantized_linear_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t input_offset,
    int64_t output_offset,
    const Tensor& multipliers,
    const Tensor& shifts,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  ESP_NN_CHECK(
      context,
      is_int8(input) && is_int8(weight) && is_int8(out) && weight.dim() == 2,
      out,
      "espressif::quantized_linear: expected int8 input, 2-D int8 weight and int8 output");
  ESP_NN_CHECK(
      context,
      is_contiguous(input) && is_contiguous(out),
      out,
      "espressif::quantized_linear: input and output must be contiguous");

  const int64_t out_features = weight.size(0);
  const int64_t in_features = weight.size(1);
  ESP_NN_CHECK(
      context,
      input.size(input.dim() - 1) == in_features &&
          out.size(out.dim() - 1) == out_features && fits_uint16(in_features) &&
          fits_uint16(out_features),
      out,
      "espressif::quantized_linear: unsupported shape");
  ESP_NN_CHECK(
      context,
      multipliers.scalar_type() == ScalarType::Int &&
          shifts.scalar_type() == ScalarType::Int &&
          multipliers.numel() == out_features && shifts.numel() == out_features,
      out,
      "espressif::quantized_linear: expected one int32 multiplier and shift per output feature");
  ESP_NN_CHECK(
      context,
      !bias.has_value() ||
          (bias->scalar_type() == ScalarType::Int &&
           bias->numel() == out_features),
      out,
      "espressif::quantized_linear: bias must be int32 with one value per output feature");

  const int64_t rows = input.numel() / in_features;
  const int8_t* in = input.const_data_ptr<int8_t>();
  int8_t* dst = out.mutable_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias->const_data_ptr<int32_t>() : nullptr;

  for (int64_t row = 0; row < rows; ++row) {
    esp_nn_fully_connected_per_ch_s8(
        in + row * in_features,
        static_cast<int32_t>(input_offset),
        static_cast<uint16_t>(in_features),
        weight.const_data_ptr<int8_t>(),
        /*filter_offset=*/0,
        bias_data,
        dst + row * out_features,
        static_cast<uint16_t>(out_features),
        static_cast<int32_t>(output_offset),
        shifts.const_data_ptr<int32_t>(),
        multipliers.const_data_ptr<int32_t>(),
        static_cast<int32_t>(activation_min),
        static_cast<int32_t>(activation_max));
  }
  return out;
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantized_linear.out",
    espressif::native::quantized_linear_out);
