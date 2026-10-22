# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
from executorch.backends.espressif.utils import (
    multiply_by_quantized_mult,
    quantize_multiplier_aot,
)

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def _c_shift_right(value: int, shift: int) -> int:
    # Arithmetic right shift; Python's >> already floors like C on two's
    # complement integers.
    return value >> shift


def _esp_nn_multiply_by_quantized_mult(x: int, mult: int, shift: int) -> int:
    """Line-by-line transcription of esp-nn's common_functions.h."""
    left_shift = shift if shift > 0 else 0
    right_shift = 0 if shift > 0 else -shift
    x = _to_int32(x * (1 << left_shift))

    overflow = x == mult == INT32_MIN
    nudge = 1 << 30
    if (x < 0) ^ (mult < 0):
        nudge = 1 - nudge
    product = x * mult + nudge
    to_add = ((1 << 31) - 1) if product < 0 else 0
    result = INT32_MAX if overflow else _to_int32((product + to_add) >> 31)

    mask = (1 << right_shift) - 1
    remainder = result & mask
    result = _c_shift_right(result, right_shift)
    threshold = (mask >> 1) + (1 if result < 0 else 0)
    return result + (1 if remainder > threshold else 0)


def test_multiply_by_quantized_mult_matches_esp_nn():
    rng = random.Random(0)
    for _ in range(2000):
        scale = 10 ** rng.uniform(-6, 0.5)
        mult, shift = quantize_multiplier_aot(scale)
        x = rng.randint(-(1 << 20), 1 << 20)
        expected = _esp_nn_multiply_by_quantized_mult(x, mult, shift)
        actual = int(multiply_by_quantized_mult(torch.tensor([x]), mult, shift)[0])
        assert actual == expected, (x, mult, shift)


def test_multiply_by_quantized_mult_per_channel():
    x = torch.arange(-50, 50).reshape(1, 4, 25)
    pairs = [quantize_multiplier_aot(s) for s in (0.01, 0.2, 0.7, 1.5)]
    mult = torch.tensor([m for m, _ in pairs]).reshape(1, 4, 1)
    shift = torch.tensor([s for _, s in pairs]).reshape(1, 4, 1)
    actual = multiply_by_quantized_mult(x, mult, shift)
    for c, (m, s) in enumerate(pairs):
        for i in range(25):
            v = int(x[0, c, i])
            assert int(actual[0, c, i]) == _esp_nn_multiply_by_quantized_mult(v, m, s)
