# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pattern checks restricting quantization to what esp-nn can execute.

A pattern that is annotated but cannot be lowered would leave a float kernel
reading int8 data, so every restriction of the device kernels is enforced
here, before annotation. Rejected patterns stay in fp32 on portable kernels.
"""

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.quantizer.arm_quantizer_utils import PatternCheck
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.cortex_m.quantizer.pattern_checkers import (
    CortexMAvgPool2DCheck,
    CortexMLinearCheck,
    CortexMMaxPool2DCheck,
    CortexMSoftmaxCheck,
)
from executorch.backends.cortex_m.utils import coerce_int_pair, is_channels_last
from torch.fx import Node

# esp-nn takes spatial sizes and channel counts as uint16_t.
_UINT16_MAX = 0xFFFF


def _fits_uint16(tensor: torch.Tensor) -> bool:
    return all(dim <= _UINT16_MAX for dim in tensor.shape)


def _arg(node: Node, index: int, default):
    value = node.args[index] if len(node.args) > index else default
    return default if value is None else value


def _symmetric_pair(value, default: tuple[int, int]) -> tuple[int, int] | None:
    pair = coerce_int_pair(value, default)
    return pair if pair is not None and len(pair) == 2 else None


class EspressifAddMulCheck(PatternCheck):
    """esp-nn add and mul are strictly elementwise: no broadcasting, and only
    an implicit alpha of one."""

    @classmethod
    def check_pattern(cls, pattern):
        for node in pattern:
            if node.kwargs.get("alpha", 1) != 1:
                return False
            if len(node.all_input_nodes) == 2:
                lhs = get_first_fake_tensor(node.all_input_nodes[0])
                rhs = get_first_fake_tensor(node.all_input_nodes[1])
                if lhs.shape != rhs.shape or lhs.stride() != rhs.stride():
                    return False
        return True

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        is_per_tensor = PatternCheck.is_per_tensor(
            quantization_config.get_input_act_qspec()
        ) and PatternCheck.is_per_tensor(quantization_config.get_output_act_qspec())
        return is_per_tensor and cls.is_int8_activations(quantization_config)


class EspressifLinearCheck(CortexMLinearCheck):
    @classmethod
    def check_pattern(cls, pattern):
        linear = pattern[0]
        weight = get_first_fake_tensor(linear.args[1])
        return weight.dim() == 2 and _fits_uint16(weight)


class EspressifConv2DCheck(PatternCheck):
    """Regular (groups == 1) and depthwise (groups == in_channels) 2D
    convolutions with unit dilation on channels-last activations."""

    @classmethod
    def check_pattern(cls, pattern):
        conv = pattern[0]
        input_tensor = get_first_fake_tensor(conv.args[0])
        weight = get_first_fake_tensor(conv.args[1])
        if input_tensor.dim() != 4 or not _fits_uint16(input_tensor):
            return False
        for node in pattern:
            if not is_channels_last(get_first_fake_tensor(node)):
                return False

        stride = _symmetric_pair(_arg(conv, 3, (1, 1)), (1, 1))
        padding = _arg(conv, 4, (0, 0))
        dilation = _symmetric_pair(_arg(conv, 5, (1, 1)), (1, 1))
        groups = _arg(conv, 6, 1)
        if isinstance(padding, str) or stride is None or dilation is None:
            return False
        if _symmetric_pair(padding, (0, 0)) is None:
            return False
        if dilation != (1, 1):
            return False

        in_channels = input_tensor.shape[1]
        out_channels = weight.shape[0]
        is_depthwise = groups == in_channels and out_channels % in_channels == 0
        return groups == 1 or is_depthwise

    @classmethod
    def check_quantization_config(
        cls, pattern: list[Node], quantization_config: QuantizationConfig
    ) -> bool:
        weight_qspec = quantization_config.get_weight_qspec(pattern[0])
        if weight_qspec is None or getattr(weight_qspec, "ch_axis", 0) not in (
            0,
            None,
        ):
            return False
        return cls.is_int8_activations(quantization_config)


def _check_pool(node: Node) -> bool:
    tensor = get_first_fake_tensor(node.args[0])
    if tensor.dim() != 4 or not is_channels_last(tensor) or not _fits_uint16(tensor):
        return False
    kernel = _symmetric_pair(node.args[1], None)
    if kernel is None:
        return False
    stride = _symmetric_pair(_arg(node, 2, kernel), kernel)
    padding = _symmetric_pair(_arg(node, 3, (0, 0)), (0, 0))
    return stride is not None and padding is not None


class EspressifMaxPool2DCheck(CortexMMaxPool2DCheck):
    @classmethod
    def check_pattern(cls, pattern):
        pool = pattern[0]
        dilation = _symmetric_pair(_arg(pool, 4, (1, 1)), (1, 1))
        ceil_mode = bool(_arg(pool, 5, False))
        return _check_pool(pool) and dilation == (1, 1) and not ceil_mode


class EspressifAvgPool2DCheck(CortexMAvgPool2DCheck):
    @classmethod
    def check_pattern(cls, pattern):
        pool = pattern[0]
        padding = _symmetric_pair(_arg(pool, 3, (0, 0)), (0, 0))
        ceil_mode = bool(_arg(pool, 4, False))
        count_include_pad = bool(_arg(pool, 5, True))
        divisor_override = _arg(pool, 6, None)
        # esp-nn averages over the in-bounds taps only.
        if padding != (0, 0) and count_include_pad:
            return False
        return _check_pool(pool) and not ceil_mode and divisor_override is None


class EspressifSoftmaxCheck(CortexMSoftmaxCheck):
    """esp_nn_softmax_s8 normalizes contiguous rows of the last dimension."""

    @classmethod
    def check_pattern(cls, pattern):
        node = pattern[0]
        tensor = get_first_fake_tensor(node)
        rank = tensor.dim()
        dim = _arg(node, 1, -1) % rank
        return tensor.is_contiguous() and dim == rank - 1
