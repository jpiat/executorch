# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Definitions of the ``espressif`` operator dialect.

Each operator maps one-to-one onto an esp-nn int8 kernel. The Python
implementations are integer-exact references of those kernels, used to check
lowered graphs on the host before running them on an ESP32-S3.

Spatial operators follow the channels-last contract of the device kernels:
activations are logically NCHW with a channels_last dim order, so the bytes in
memory are NHWC. Convolution weights are repacked ahead of time into the
layout esp-nn reads.

Offsets are the negated zero points, matching esp-nn's ``in_offset`` and
``out_offset`` arguments: a kernel computes ``(x + input_offset) * w``.
"""

import math

import torch
import torch.nn.functional as F
from executorch.backends.espressif.utils import (
    ADD_LEFT_SHIFT,
    multiply_by_quantized_mult,
    SOFTMAX_INPUT_INTEGER_BITS,
    SOFTMAX_OUTPUT_SCALE,
    SOFTMAX_OUTPUT_ZERO_POINT,
)
from torch.library import impl, Library, register_fake

lib = Library("espressif", "DEF")


def _define(schema: str) -> None:
    """Define a functional operator and its out variant from one schema."""
    name, rest = schema.split("(", 1)
    arguments, returns = rest.rsplit(") -> ", 1)
    lib.define(schema)
    lib.define(f"{name}.out({arguments}, *, Tensor(a!) out) -> Tensor(a!)")


def _requantize_and_clamp(
    acc: torch.Tensor,
    multipliers,
    shifts,
    output_offset: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    result = multiply_by_quantized_mult(acc, multipliers, shifts) + output_offset
    return torch.clamp(result, activation_min, activation_max).to(torch.int8)


def _channels_last_like(shape, device) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int8, device=device).to(
        memory_format=torch.channels_last
    )


def _conv_output_hw(input_hw, kernel_hw, stride, padding):
    return [
        (input_hw[i] + 2 * padding[i] - kernel_hw[i]) // stride[i] + 1 for i in range(2)
    ]


###
# quantize_per_tensor / dequantize_per_tensor
###

_define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@register_fake("espressif::quantize_per_tensor")
def quantize_per_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.empty_like(input, dtype=dtype)


@impl(lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_impl(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.ops.quantized_decomposed.quantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


_define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@register_fake("espressif::dequantize_per_tensor")
def dequantize_per_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.empty_like(input, dtype=torch.float)


@impl(lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_impl(input, scale, zero_point, quant_min, quant_max, dtype):
    return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


###
# quantized_linear -> esp_nn_fully_connected_per_ch_s8
###

_define(
    "quantized_linear(Tensor input, Tensor weight, Tensor? bias, "
    "int input_offset, int output_offset, Tensor multipliers, Tensor shifts, "
    "int activation_min, int activation_max) -> Tensor"
)


@register_fake("espressif::quantized_linear")
def quantized_linear_meta(
    input,
    weight,
    bias,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    return torch.empty(
        (*input.shape[:-1], weight.shape[0]), dtype=torch.int8, device=input.device
    )


@impl(lib, "quantized_linear", "CompositeExplicitAutograd")
def quantized_linear_impl(
    input,
    weight,
    bias,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    rows = input.to(torch.int64).reshape(-1, input.shape[-1]) + input_offset
    acc = rows @ weight.to(torch.int64).T
    if bias is not None:
        acc = acc + bias.to(torch.int64)
    out = _requantize_and_clamp(
        acc, multipliers, shifts, output_offset, activation_min, activation_max
    )
    return out.reshape(*input.shape[:-1], weight.shape[0])


###
# quantized_conv2d -> esp_nn_conv_s8
#
# weight: [out_channels, kernel_h, kernel_w, in_channels] (OHWI), contiguous.
###

_define(
    "quantized_conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, "
    "int[] padding, int input_offset, int output_offset, Tensor multipliers, "
    "Tensor shifts, int activation_min, int activation_max) -> Tensor"
)


@register_fake("espressif::quantized_conv2d")
def quantized_conv2d_meta(
    input,
    weight,
    bias,
    stride,
    padding,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    out_hw = _conv_output_hw(input.shape[2:], weight.shape[1:3], stride, padding)
    return _channels_last_like((input.shape[0], weight.shape[0], *out_hw), input.device)


def _integer_conv2d(input, weight_oihw, bias, stride, padding, groups, input_offset):
    # Products of int8 values summed in float64 stay exact for any realistic
    # kernel volume, and zero padding of (x + offset) is what esp-nn computes
    # when it skips out-of-bounds taps.
    acc = F.conv2d(
        input.to(torch.float64) + input_offset,
        weight_oihw.to(torch.float64),
        None,
        stride,
        padding,
        1,
        groups,
    ).to(torch.int64)
    if bias is not None:
        acc = acc + bias.to(torch.int64).reshape(1, -1, 1, 1)
    return acc


@impl(lib, "quantized_conv2d", "CompositeExplicitAutograd")
def quantized_conv2d_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    acc = _integer_conv2d(
        input, weight.permute(0, 3, 1, 2), bias, stride, padding, 1, input_offset
    )
    out = _requantize_and_clamp(
        acc,
        multipliers.reshape(1, -1, 1, 1),
        shifts.reshape(1, -1, 1, 1),
        output_offset,
        activation_min,
        activation_max,
    )
    return out.contiguous(memory_format=torch.channels_last)


###
# quantized_depthwise_conv2d -> esp_nn_depthwise_conv_s8
#
# weight: [1, kernel_h, kernel_w, out_channels] (IHWO), contiguous, with
# out_channels = in_channels * depth_multiplier.
###

_define(
    "quantized_depthwise_conv2d(Tensor input, Tensor weight, Tensor? bias, "
    "int[] stride, int[] padding, int depth_multiplier, int input_offset, "
    "int output_offset, Tensor multipliers, Tensor shifts, int activation_min, "
    "int activation_max) -> Tensor"
)


@register_fake("espressif::quantized_depthwise_conv2d")
def quantized_depthwise_conv2d_meta(
    input,
    weight,
    bias,
    stride,
    padding,
    depth_multiplier,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    out_hw = _conv_output_hw(input.shape[2:], weight.shape[1:3], stride, padding)
    return _channels_last_like((input.shape[0], weight.shape[3], *out_hw), input.device)


@impl(lib, "quantized_depthwise_conv2d", "CompositeExplicitAutograd")
def quantized_depthwise_conv2d_impl(
    input,
    weight,
    bias,
    stride,
    padding,
    depth_multiplier,
    input_offset,
    output_offset,
    multipliers,
    shifts,
    activation_min,
    activation_max,
):
    in_channels = input.shape[1]
    acc = _integer_conv2d(
        input,
        weight.permute(3, 0, 1, 2),
        bias,
        stride,
        padding,
        in_channels,
        input_offset,
    )
    out = _requantize_and_clamp(
        acc,
        multipliers.reshape(1, -1, 1, 1),
        shifts.reshape(1, -1, 1, 1),
        output_offset,
        activation_min,
        activation_max,
    )
    return out.contiguous(memory_format=torch.channels_last)


###
# quantized_add -> esp_nn_add_elementwise_s8
###

_define(
    "quantized_add(Tensor self, Tensor other, int self_offset, "
    "int self_multiplier, int self_shift, int other_offset, "
    "int other_multiplier, int other_shift, int output_offset, "
    "int output_multiplier, int output_shift, int activation_min, "
    "int activation_max) -> Tensor"
)


@register_fake("espressif::quantized_add")
def quantized_add_meta(self, other, *args):
    return torch.empty_like(self)


@impl(lib, "quantized_add", "CompositeExplicitAutograd")
def quantized_add_impl(
    self,
    other,
    self_offset,
    self_multiplier,
    self_shift,
    other_offset,
    other_multiplier,
    other_shift,
    output_offset,
    output_multiplier,
    output_shift,
    activation_min,
    activation_max,
):
    lhs = (self.to(torch.int64) + self_offset) << ADD_LEFT_SHIFT
    rhs = (other.to(torch.int64) + other_offset) << ADD_LEFT_SHIFT
    acc = multiply_by_quantized_mult(
        lhs, self_multiplier, self_shift
    ) + multiply_by_quantized_mult(rhs, other_multiplier, other_shift)
    out = _requantize_and_clamp(
        acc,
        output_multiplier,
        output_shift,
        output_offset,
        activation_min,
        activation_max,
    )
    return out.contiguous(memory_format=_memory_format(self))


###
# quantized_mul -> esp_nn_mul_elementwise_s8
###

_define(
    "quantized_mul(Tensor self, Tensor other, int self_offset, int other_offset, "
    "int output_offset, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)


@register_fake("espressif::quantized_mul")
def quantized_mul_meta(self, other, *args):
    return torch.empty_like(self)


@impl(lib, "quantized_mul", "CompositeExplicitAutograd")
def quantized_mul_impl(
    self,
    other,
    self_offset,
    other_offset,
    output_offset,
    output_multiplier,
    output_shift,
    activation_min,
    activation_max,
):
    acc = (self.to(torch.int64) + self_offset) * (other.to(torch.int64) + other_offset)
    out = _requantize_and_clamp(
        acc,
        output_multiplier,
        output_shift,
        output_offset,
        activation_min,
        activation_max,
    )
    return out.contiguous(memory_format=_memory_format(self))


def _memory_format(tensor: torch.Tensor) -> torch.memory_format:
    if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    return torch.contiguous_format


###
# quantized_max_pool2d -> esp_nn_max_pool_s8
# quantized_avg_pool2d -> esp_nn_avg_pool_s8
#
# Input and output share quantization parameters, so only the clamp range
# of a fused activation is carried.
###

_define(
    "quantized_max_pool2d(Tensor input, int[] kernel_size, int[] stride, "
    "int[] padding, int activation_min, int activation_max) -> Tensor"
)
_define(
    "quantized_avg_pool2d(Tensor input, int[] kernel_size, int[] stride, "
    "int[] padding, int activation_min, int activation_max) -> Tensor"
)


def _pool_meta(input, kernel_size, stride, padding, activation_min, activation_max):
    out_hw = _conv_output_hw(input.shape[2:], kernel_size, stride, padding)
    return _channels_last_like((*input.shape[:2], *out_hw), input.device)


register_fake("espressif::quantized_max_pool2d")(_pool_meta)
register_fake("espressif::quantized_avg_pool2d")(_pool_meta)


@impl(lib, "quantized_max_pool2d", "CompositeExplicitAutograd")
def quantized_max_pool2d_impl(
    input, kernel_size, stride, padding, activation_min, activation_max
):
    # esp-nn ignores out-of-bounds taps, which -inf padding reproduces.
    padded = F.pad(
        input.to(torch.float64),
        (padding[1], padding[1], padding[0], padding[0]),
        value=-math.inf,
    )
    out = F.max_pool2d(padded, kernel_size, stride)
    out = torch.clamp(out, activation_min, activation_max).to(torch.int8)
    return out.contiguous(memory_format=torch.channels_last)


@impl(lib, "quantized_avg_pool2d", "CompositeExplicitAutograd")
def quantized_avg_pool2d_impl(
    input, kernel_size, stride, padding, activation_min, activation_max
):
    volume = kernel_size[0] * kernel_size[1]
    as_double = input.to(torch.float64)
    total = F.avg_pool2d(as_double, kernel_size, stride, padding) * volume
    count = (
        F.avg_pool2d(torch.ones_like(as_double), kernel_size, stride, padding) * volume
    )
    total = torch.round(total).to(torch.int64)
    count = torch.round(count).to(torch.int64)
    # esp-nn divides by the number of in-bounds taps and rounds half away
    # from zero using C integer division.
    half = count // 2
    out = torch.where(
        total > 0,
        torch.div(total + half, count, rounding_mode="trunc"),
        torch.div(total - half, count, rounding_mode="trunc"),
    )
    out = torch.clamp(out, activation_min, activation_max).to(torch.int8)
    return out.contiguous(memory_format=torch.channels_last)


###
# quantized_softmax -> esp_nn_softmax_s8 over the last dimension
#
# Output quantization is fixed to scale 1/256 and zero point -128.
###

_define(
    "quantized_softmax(Tensor input, int input_multiplier, int input_shift, "
    "int diff_min) -> Tensor"
)


@register_fake("espressif::quantized_softmax")
def quantized_softmax_meta(input, input_multiplier, input_shift, diff_min):
    return torch.empty_like(input, dtype=torch.int8)


@impl(lib, "quantized_softmax", "CompositeExplicitAutograd")
def quantized_softmax_impl(input, input_multiplier, input_shift, diff_min):
    # A float reference: the device evaluates exp() with a fixed-point
    # approximation, so outputs agree to about one quantization step.
    input_scale = math.ldexp(input_multiplier / float(1 << 31), input_shift) / float(
        1 << (31 - SOFTMAX_INPUT_INTEGER_BITS)
    )
    diff = input.to(torch.int32) - input.to(torch.int32).amax(dim=-1, keepdim=True)
    probs = torch.softmax(diff.to(torch.float64) * input_scale, dim=-1)
    probs = torch.where(diff >= diff_min, probs, torch.zeros_like(probs))
    out = torch.round(probs / SOFTMAX_OUTPUT_SCALE) + SOFTMAX_OUTPUT_ZERO_POINT
    return torch.clamp(out, -128, 127).to(torch.int8)
