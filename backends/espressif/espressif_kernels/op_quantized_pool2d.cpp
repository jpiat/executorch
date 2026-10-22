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

using PoolFn = void (*)(
    const int8_t*,
    uint16_t,
    uint16_t,
    int8_t*,
    uint16_t,
    uint16_t,
    uint16_t,
    uint16_t,
    uint16_t,
    uint16_t,
    uint16_t,
    uint16_t,
    int32_t,
    int32_t,
    uint16_t);

Tensor& run_pool(
    KernelRuntimeContext& context,
    const char* op,
    PoolFn pool,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  ESP_NN_CHECK(
      context,
      is_int8(input) && is_int8(out) && is_nhwc(input) && is_nhwc(out),
      out,
      "%s: expected int8 tensors with channels_last dim order",
      op);
  ESP_NN_CHECK(
      context,
      kernel_size.size() == 2 && stride.size() == 2 && padding.size() == 2,
      out,
      "%s: kernel_size, stride and padding must have two values",
      op);
  ESP_NN_CHECK(
      context,
      fits_uint16(input.size(1)) && fits_uint16(input.size(2)) &&
          fits_uint16(input.size(3)),
      out,
      "%s: dimensions exceed esp-nn's 16-bit limits",
      op);

  const int64_t channels = input.size(1);
  const int64_t input_batch_stride = input.numel() / input.size(0);
  const int64_t output_batch_stride = out.numel() / out.size(0);
  for (int64_t n = 0; n < input.size(0); ++n) {
    pool(
        input.const_data_ptr<int8_t>() + n * input_batch_stride,
        static_cast<uint16_t>(input.size(3)),
        static_cast<uint16_t>(input.size(2)),
        out.mutable_data_ptr<int8_t>() + n * output_batch_stride,
        static_cast<uint16_t>(out.size(3)),
        static_cast<uint16_t>(out.size(2)),
        static_cast<uint16_t>(stride[1]),
        static_cast<uint16_t>(stride[0]),
        static_cast<uint16_t>(kernel_size[1]),
        static_cast<uint16_t>(kernel_size[0]),
        static_cast<uint16_t>(padding[1]),
        static_cast<uint16_t>(padding[0]),
        static_cast<int32_t>(activation_min),
        static_cast<int32_t>(activation_max),
        static_cast<uint16_t>(channels));
  }
  return out;
}

} // namespace

Tensor& quantized_max_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  return run_pool(
      context,
      "espressif::quantized_max_pool2d",
      esp_nn_max_pool_s8,
      input,
      kernel_size,
      stride,
      padding,
      activation_min,
      activation_max,
      out);
}

Tensor& quantized_avg_pool2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t activation_min,
    int64_t activation_max,
    Tensor& out) {
  return run_pool(
      context,
      "espressif::quantized_avg_pool2d",
      esp_nn_avg_pool_s8,
      input,
      kernel_size,
      stride,
      padding,
      activation_min,
      activation_max,
      out);
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantized_max_pool2d.out",
    espressif::native::quantized_max_pool2d_out);
EXECUTORCH_LIBRARY(
    espressif,
    "quantized_avg_pool2d.out",
    espressif::native::quantized_avg_pool2d_out);
