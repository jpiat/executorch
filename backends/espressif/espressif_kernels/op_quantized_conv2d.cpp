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

// Shared validation of conv-like arguments. Returns false after failing the
// context.
bool check_conv_args(
    KernelRuntimeContext& context,
    const char* op,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    const Tensor& multipliers,
    const Tensor& shifts,
    const Tensor& out) {
  if (!is_int8(input) || !is_int8(weight) || !is_int8(out) ||
      weight.dim() != 4) {
    ET_LOG(Error, "%s: expected int8 tensors and a 4-D weight", op);
  } else if (!is_nhwc(input) || !is_nhwc(out)) {
    ET_LOG(Error, "%s: input and output must have channels_last dim order", op);
  } else if (stride.size() != 2 || padding.size() != 2) {
    ET_LOG(Error, "%s: stride and padding must have two values", op);
  } else if (
      multipliers.scalar_type() != ScalarType::Int ||
      shifts.scalar_type() != ScalarType::Int ||
      multipliers.numel() != out.size(1) || shifts.numel() != out.size(1)) {
    ET_LOG(Error, "%s: expected int32 multiplier and shift per channel", op);
  } else if (
      bias.has_value() &&
      (bias->scalar_type() != ScalarType::Int ||
       bias->numel() != out.size(1))) {
    ET_LOG(Error, "%s: bias must be int32 with one value per channel", op);
  } else if (
      !fits_uint16(input.size(1)) || !fits_uint16(input.size(2)) ||
      !fits_uint16(input.size(3)) || !fits_uint16(out.size(1))) {
    ET_LOG(Error, "%s: dimensions exceed esp-nn's 16-bit limits", op);
  } else {
    return true;
  }
  context.fail(Error::InvalidArgument);
  return false;
}

data_dims_t nhwc_dims(const Tensor& t) {
  return data_dims_t{
      static_cast<int32_t>(t.size(3)),
      static_cast<int32_t>(t.size(2)),
      static_cast<int32_t>(t.size(1)),
      1};
}

} // namespace

Tensor& quantized_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t input_offset,
    int64_t output_offset,
    const Tensor& multipliers,
    const Tensor& shifts,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  if (!check_conv_args(
          context,
          "espressif::quantized_conv2d",
          input,
          weight,
          bias,
          stride,
          padding,
          multipliers,
          shifts,
          out)) {
    return out;
  }
  // weight is OHWI.
  ESP_NN_CHECK(
      context,
      weight.size(0) == out.size(1) && weight.size(3) == input.size(1),
      out,
      "espressif::quantized_conv2d: weight shape does not match channels");

  const data_dims_t input_dims = nhwc_dims(input);
  const data_dims_t output_dims = nhwc_dims(out);
  const data_dims_t filter_dims{
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(weight.size(1)),
      static_cast<int32_t>(weight.size(3)),
      static_cast<int32_t>(weight.size(0))};
  const conv_params_t params{
      static_cast<int32_t>(input_offset),
      static_cast<int32_t>(output_offset),
      {static_cast<int32_t>(stride[1]), static_cast<int32_t>(stride[0])},
      {static_cast<int32_t>(padding[1]), static_cast<int32_t>(padding[0])},
      {1, 1},
      {static_cast<int32_t>(activation_min),
       static_cast<int32_t>(activation_max)}};
  const quant_data_t quant{
      const_cast<int32_t*>(shifts.const_data_ptr<int32_t>()),
      const_cast<int32_t*>(multipliers.const_data_ptr<int32_t>())};

  void* scratch = nullptr;
  if (!allocate_scratch(
          context,
          esp_nn_get_conv_scratch_size(
              &input_dims, &filter_dims, &output_dims, &params),
          &scratch)) {
    return out;
  }

  const int64_t input_batch_stride = input.numel() / input.size(0);
  const int64_t output_batch_stride = out.numel() / out.size(0);
  const int32_t* bias_data =
      bias.has_value() ? bias->const_data_ptr<int32_t>() : nullptr;
  for (int64_t n = 0; n < input.size(0); ++n) {
    esp_nn_set_conv_scratch_buf(scratch);
    esp_nn_conv_s8(
        &input_dims,
        input.const_data_ptr<int8_t>() + n * input_batch_stride,
        &filter_dims,
        weight.const_data_ptr<int8_t>(),
        bias_data,
        &output_dims,
        out.mutable_data_ptr<int8_t>() + n * output_batch_stride,
        &params,
        &quant);
  }
  return out;
}

Tensor& quantized_depthwise_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t depth_multiplier,
    int64_t input_offset,
    int64_t output_offset,
    const Tensor& multipliers,
    const Tensor& shifts,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  if (!check_conv_args(
          context,
          "espressif::quantized_depthwise_conv2d",
          input,
          weight,
          bias,
          stride,
          padding,
          multipliers,
          shifts,
          out)) {
    return out;
  }
  // weight is 1HWO with O = C * depth_multiplier.
  ESP_NN_CHECK(
      context,
      weight.size(0) == 1 && weight.size(3) == out.size(1) &&
          out.size(1) == input.size(1) * depth_multiplier,
      out,
      "espressif::quantized_depthwise_conv2d: weight shape does not match channels");

  const data_dims_t input_dims = nhwc_dims(input);
  const data_dims_t output_dims = nhwc_dims(out);
  const data_dims_t filter_dims{
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(weight.size(1)),
      static_cast<int32_t>(weight.size(3)),
      1};
  const dw_conv_params_t params{
      static_cast<int32_t>(input_offset),
      static_cast<int32_t>(output_offset),
      static_cast<int32_t>(depth_multiplier),
      {static_cast<int32_t>(stride[1]), static_cast<int32_t>(stride[0])},
      {static_cast<int32_t>(padding[1]), static_cast<int32_t>(padding[0])},
      {1, 1},
      {static_cast<int32_t>(activation_min),
       static_cast<int32_t>(activation_max)}};
  const quant_data_t quant{
      const_cast<int32_t*>(shifts.const_data_ptr<int32_t>()),
      const_cast<int32_t*>(multipliers.const_data_ptr<int32_t>())};

  void* scratch = nullptr;
  if (!allocate_scratch(
          context,
          esp_nn_get_depthwise_conv_scratch_size(
              &input_dims, &filter_dims, &output_dims, &params),
          &scratch)) {
    return out;
  }

  const int64_t input_batch_stride = input.numel() / input.size(0);
  const int64_t output_batch_stride = out.numel() / out.size(0);
  const int32_t* bias_data =
      bias.has_value() ? bias->const_data_ptr<int32_t>() : nullptr;
  for (int64_t n = 0; n < input.size(0); ++n) {
    esp_nn_set_depthwise_conv_scratch_buf(scratch);
    esp_nn_depthwise_conv_s8(
        &input_dims,
        input.const_data_ptr<int8_t>() + n * input_batch_stride,
        &filter_dims,
        weight.const_data_ptr<int8_t>(),
        bias_data,
        &output_dims,
        out.mutable_data_ptr<int8_t>() + n * output_batch_stride,
        &params,
        &quant);
  }
  return out;
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantized_conv2d.out",
    espressif::native::quantized_conv2d_out);
EXECUTORCH_LIBRARY(
    espressif,
    "quantized_depthwise_conv2d.out",
    espressif::native::quantized_depthwise_conv2d_out);
