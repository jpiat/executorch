# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integer arithmetic shared by the AOT passes and the reference operators.

esp-nn follows the TensorFlow Lite Micro int8 scheme: a real rescale factor is
encoded as a Q31 multiplier and a power-of-two shift, positive shifts moving
left. The helpers below mirror esp-nn's ``common_functions.h`` so the Python
reference operators match the device kernels to within rounding of the
optimized assembly paths.
"""

import torch

from executorch.backends.cortex_m.utils import quantize_multiplier_aot

# Pre-shift applied to both add operands before rescaling; the value used by
# TFLite Micro and expected by esp_nn_add_elementwise_s8.
ADD_LEFT_SHIFT = 20

# esp_nn_softmax_s8 always produces this output quantization.
SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128

# Q5.26 input representation used by the softmax exp() approximation.
SOFTMAX_INPUT_INTEGER_BITS = 5

__all__ = [
    "ADD_LEFT_SHIFT",
    "SOFTMAX_INPUT_INTEGER_BITS",
    "SOFTMAX_OUTPUT_SCALE",
    "SOFTMAX_OUTPUT_ZERO_POINT",
    "div_by_power_of_two",
    "multiply_by_quantized_mult",
    "quantize_multiplier_aot",
    "sat_round_doubling_high_mul",
]


def sat_round_doubling_high_mul(x: torch.Tensor, multiplier) -> torch.Tensor:
    """Mirror esp_nn_sat_round_doubling_high_mul on int64 tensors."""
    x = x.to(torch.int64)
    multiplier = torch.as_tensor(multiplier, dtype=torch.int64)
    nudge = torch.where((x < 0) ^ (multiplier < 0), 1 - (1 << 30), 1 << 30)
    product = x * multiplier + nudge
    # Add 2^31 - 1 to negative products so the arithmetic shift truncates
    # toward zero, as esp_nn_pick_sat_high32_of64 does.
    product = product + torch.where(product < 0, (1 << 31) - 1, 0)
    return product >> 31


def div_by_power_of_two(x: torch.Tensor, exponent) -> torch.Tensor:
    """Mirror esp_nn_div_by_power_of_two: divide by 2^exponent rounding half
    away from zero."""
    x = x.to(torch.int64)
    exponent = torch.as_tensor(exponent, dtype=torch.int64)
    mask = (torch.ones_like(exponent) << exponent) - 1
    remainder = x & mask
    result = x >> exponent
    threshold = (mask >> 1) + (result < 0).to(torch.int64)
    return result + (remainder > threshold).to(torch.int64)


def multiply_by_quantized_mult(x: torch.Tensor, multiplier, shift) -> torch.Tensor:
    """Mirror esp_nn_multiply_by_quantized_mult.

    ``multiplier`` and ``shift`` may be scalars or tensors broadcastable to
    ``x`` (per-channel requantization).
    """
    shift = torch.as_tensor(shift, dtype=torch.int64)
    left_shift = torch.clamp(shift, min=0)
    right_shift = torch.clamp(-shift, min=0)
    scaled = x.to(torch.int64) * (torch.ones_like(left_shift) << left_shift)
    return div_by_power_of_two(
        sat_round_doubling_high_mul(scaled, multiplier), right_shift
    )
