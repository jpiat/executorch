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

// Softmax over the last dimension. The output is quantized with scale 1/256
// and zero point -128.
Tensor& quantized_softmax_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t input_multiplier,
    int64_t input_shift,
    int64_t diff_min,
    Tensor& out) {
  ESP_NN_CHECK(
      context,
      is_int8(input) && is_int8(out) && input.numel() == out.numel() &&
          input.dim() > 0 && is_contiguous(input) && is_contiguous(out),
      out,
      "espressif::quantized_softmax: expected contiguous int8 tensors");

  const int32_t width = static_cast<int32_t>(input.size(input.dim() - 1));
  const int32_t height = static_cast<int32_t>(input.numel() / width);

  void* scratch = nullptr;
  if (!allocate_scratch(
          context, esp_nn_get_softmax_scratch_size(width, height), &scratch)) {
    return out;
  }
  esp_nn_set_softmax_scratch_buf(scratch);
  esp_nn_softmax_s8(
      input.const_data_ptr<int8_t>(),
      height,
      width,
      static_cast<int32_t>(input_multiplier),
      static_cast<int32_t>(input_shift),
      static_cast<int32_t>(diff_min),
      out.mutable_data_ptr<int8_t>());
  return out;
}

} // namespace native
} // namespace espressif

EXECUTORCH_LIBRARY(
    espressif,
    "quantized_softmax.out",
    espressif::native::quantized_softmax_out);
