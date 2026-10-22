# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace quantized ATen operators with esp-nn backed ``espressif`` ops.

Runs after FoldAndAnnotateQParamsPass, which leaves quantized operators
reading and writing int8 tensors with their quantization parameters in
``node.meta``. Each substitution turns those parameters into the fixed-point
arguments of the corresponding esp-nn kernel.

Every node reaching a substitution with qparams has already lost its
dequantize/quantize neighbours, so a substitution that cannot lower such a
node raises instead of returning None: leaving it would run a float kernel on
int8 data.
"""

import math
from typing import cast

import executorch.backends.espressif.ops.operators  # noqa: F401
import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.espressif.utils import (
    ADD_LEFT_SHIFT,
    quantize_multiplier_aot,
    SOFTMAX_INPUT_INTEGER_BITS,
    SOFTMAX_OUTPUT_SCALE,
    SOFTMAX_OUTPUT_ZERO_POINT,
)
from executorch.backends.transforms.aten_to_dialect_pass import (
    AtenToDialectPass,
    DialectNodeSpec,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx import Node


class AtenToEspressifPass(AtenToDialectPass):
    """Lower quantized ATen operators to the ``espressif`` dialect."""


def _has_qparams(node: Node) -> bool:
    return bool(node.meta.get("input_qparams")) and bool(
        node.meta.get("output_qparams")
    )


def _pair(value, default) -> list[int]:
    if value is None:
        return list(default)
    values = [int(v) for v in value]
    return values * 2 if len(values) == 1 else values[:2]


def _per_channel_requantization(
    input_scale: float, weight_scales, output_scale: float, channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(weight_scales, (list, tuple)):
        weight_scales = [weight_scales] * channels
    pairs = [
        quantize_multiplier_aot(input_scale * float(w) / output_scale)
        for w in weight_scales
    ]
    multipliers = torch.tensor([m for m, _ in pairs], dtype=torch.int32)
    shifts = torch.tensor([s for _, s in pairs], dtype=torch.int32)
    return multipliers, shifts


def _add_constant(
    dialect_pass: AtenToDialectPass, node: Node, anchor: Node, suffix: str, data
) -> Node:
    with node.graph.inserting_after(anchor):
        return create_constant_placeholder(
            dialect_pass.exported_program,
            node.graph,
            f"{node.name}_{suffix}",
            InputKind.PARAMETER,
            data,
        )


def _output_range(node: Node) -> tuple[int, int, int, float]:
    output_qparams = node.meta["output_qparams"][0]
    return (
        int(output_qparams.zp),
        int(output_qparams.qmin),
        int(output_qparams.qmax),
        float(output_qparams.scale),
    )


@AtenToEspressifPass.register_dialect_substitution(exir_ops.edge.aten.linear.default)
def _get_linear_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    if not _has_qparams(node):
        return None

    weight = cast(Node, node.args[1])
    bias = cast(Node | None, node.args[2]) if len(node.args) > 2 else None
    input_qparams = node.meta["input_qparams"][0]
    weight_qparams = node.meta["input_qparams"][1]
    if weight_qparams.zp not in (0, [0] * len(weight_qparams.zp or [])):
        raise RuntimeError(f"{node.name}: esp-nn requires symmetric weights.")
    output_zp, output_min, output_max, output_scale = _output_range(node)

    weight_tensor = get_param_tensor(dialect_pass.exported_program, weight)
    if weight_tensor is None:
        raise RuntimeError(f"{node.name}: linear weight must be a constant.")
    multipliers, shifts = _per_channel_requantization(
        float(input_qparams.scale),
        weight_qparams.scale,
        output_scale,
        weight_tensor.shape[0],
    )
    multipliers_node = _add_constant(
        dialect_pass, node, weight, "multipliers", multipliers
    )
    shifts_node = _add_constant(dialect_pass, node, weight, "shifts", shifts)

    args = (
        node.args[0],
        weight,
        bias,
        -int(input_qparams.zp),
        output_zp,
        multipliers_node,
        shifts_node,
        output_min,
        output_max,
    )
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_linear.default, args)


@AtenToEspressifPass.register_dialect_substitution(
    exir_ops.edge.aten.convolution.default
)
def _get_convolution_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    if not _has_qparams(node):
        return None

    x, weight, bias, stride, padding, dilation, transposed, _, groups = node.args
    weight = cast(Node, weight)
    if transposed or _pair(dilation, (1, 1)) != [1, 1]:
        raise RuntimeError(
            f"{node.name}: esp-nn supports neither transposed nor dilated "
            "convolutions; the quantizer should not have annotated it."
        )

    input_qparams = node.meta["input_qparams"][0]
    weight_qparams = node.meta["input_qparams"][1]
    output_zp, output_min, output_max, output_scale = _output_range(node)

    weight_tensor = get_param_tensor(dialect_pass.exported_program, weight)
    if weight_tensor is None:
        raise RuntimeError(f"{node.name}: convolution weight must be a constant.")
    out_channels = weight_tensor.shape[0]
    in_channels = weight_tensor.shape[1] * int(groups)
    multipliers, shifts = _per_channel_requantization(
        float(input_qparams.scale), weight_qparams.scale, output_scale, out_channels
    )

    is_depthwise = groups == in_channels and groups > 1
    if not is_depthwise and groups != 1:
        raise RuntimeError(f"{node.name}: grouped convolutions are not supported.")

    # esp-nn reads OHWI filters for regular convolutions and 1HWO filters,
    # channel-multiplier minor, for depthwise ones.
    if is_depthwise:
        packed = weight_tensor.permute(1, 2, 3, 0).contiguous()
    else:
        packed = weight_tensor.permute(0, 2, 3, 1).contiguous()

    packed_node = _add_constant(dialect_pass, node, weight, "weight_packed", packed)
    multipliers_node = _add_constant(
        dialect_pass, node, weight, "multipliers", multipliers
    )
    shifts_node = _add_constant(dialect_pass, node, weight, "shifts", shifts)

    common = (
        x,
        packed_node,
        bias,
        _pair(stride, (1, 1)),
        _pair(padding, (0, 0)),
    )
    quant = (
        -int(input_qparams.zp),
        output_zp,
        multipliers_node,
        shifts_node,
        output_min,
        output_max,
    )
    if is_depthwise:
        return DialectNodeSpec(
            exir_ops.edge.espressif.quantized_depthwise_conv2d.default,
            common + (out_channels // in_channels,) + quant,
        )
    return DialectNodeSpec(
        exir_ops.edge.espressif.quantized_conv2d.default, common + quant
    )


@AtenToEspressifPass.register_dialect_substitution(exir_ops.edge.aten.add.Tensor)
def _get_add_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None
    if node.kwargs.get("alpha", 1) != 1:
        raise RuntimeError(f"{node.name}: esp-nn add does not support alpha.")

    lhs_qparams = node.meta["input_qparams"][0]
    rhs_qparams = node.meta["input_qparams"][1]
    output_zp, output_min, output_max, output_scale = _output_range(node)

    # TFLite Micro add: both operands are pre-shifted by ADD_LEFT_SHIFT and
    # rescaled to twice the larger input scale, then the sum is rescaled to
    # the output scale. esp-nn applies all three shifts as right shifts.
    lhs_scale = float(lhs_qparams.scale)
    rhs_scale = float(rhs_qparams.scale)
    twice_max_scale = 2 * max(lhs_scale, rhs_scale)
    lhs_multiplier, lhs_shift = quantize_multiplier_aot(lhs_scale / twice_max_scale)
    rhs_multiplier, rhs_shift = quantize_multiplier_aot(rhs_scale / twice_max_scale)
    output_multiplier, output_shift = quantize_multiplier_aot(
        twice_max_scale / ((1 << ADD_LEFT_SHIFT) * output_scale)
    )
    if output_shift > 0:
        raise RuntimeError(
            f"{node.name}: output scale {output_scale} is too small for operand "
            f"scales {lhs_scale} and {rhs_scale}; esp-nn add can only scale "
            "the sum down. Calibrate on data where the operands do not cancel."
        )

    args = (
        node.args[0],
        node.args[1],
        -int(lhs_qparams.zp),
        lhs_multiplier,
        lhs_shift,
        -int(rhs_qparams.zp),
        rhs_multiplier,
        rhs_shift,
        output_zp,
        output_multiplier,
        output_shift,
        output_min,
        output_max,
    )
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_add.default, args)


@AtenToEspressifPass.register_dialect_substitution(exir_ops.edge.aten.mul.Tensor)
def _get_mul_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None

    lhs_qparams = node.meta["input_qparams"][0]
    rhs_qparams = node.meta["input_qparams"][1]
    output_zp, output_min, output_max, output_scale = _output_range(node)
    output_multiplier, output_shift = quantize_multiplier_aot(
        float(lhs_qparams.scale) * float(rhs_qparams.scale) / output_scale
    )
    args = (
        node.args[0],
        node.args[1],
        -int(lhs_qparams.zp),
        -int(rhs_qparams.zp),
        output_zp,
        output_multiplier,
        output_shift,
        output_min,
        output_max,
    )
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_mul.default, args)


def _pool_args(node: Node) -> tuple[list[int], list[int], list[int]]:
    kernel_size = _pair(node.args[1], None)
    stride = node.args[2] if len(node.args) > 2 and node.args[2] else kernel_size
    padding = node.args[3] if len(node.args) > 3 else None
    return kernel_size, _pair(stride, kernel_size), _pair(padding, (0, 0))


def _check_shared_qparams(node: Node) -> tuple[int, int]:
    input_qparams = node.meta["input_qparams"][0]
    output_qparams = node.meta["output_qparams"][0]
    if not math.isclose(
        float(input_qparams.scale), float(output_qparams.scale), rel_tol=1e-6
    ) or int(input_qparams.zp) != int(output_qparams.zp):
        raise RuntimeError(
            f"{node.name}: esp-nn pooling requires matching input and output "
            "quantization."
        )
    return int(output_qparams.qmin), int(output_qparams.qmax)


@AtenToEspressifPass.register_dialect_substitution(
    exir_ops.edge.aten.max_pool2d.default
)
def _get_max_pool2d_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None
    activation_min, activation_max = _check_shared_qparams(node)
    kernel_size, stride, padding = _pool_args(node)
    args = (node.args[0], kernel_size, stride, padding, activation_min, activation_max)
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_max_pool2d.default, args)


@AtenToEspressifPass.register_dialect_substitution(
    exir_ops.edge.aten.avg_pool2d.default
)
def _get_avg_pool2d_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None
    activation_min, activation_max = _check_shared_qparams(node)
    kernel_size, stride, padding = _pool_args(node)
    args = (node.args[0], kernel_size, stride, padding, activation_min, activation_max)
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_avg_pool2d.default, args)


def _softmax_parameters(input_scale: float) -> tuple[int, int, int]:
    """TFLite Micro's PreprocessSoftmaxScaling with beta = 1."""
    real_multiplier = min(
        input_scale * (1 << (31 - SOFTMAX_INPUT_INTEGER_BITS)), float((1 << 31) - 1)
    )
    multiplier, shift = quantize_multiplier_aot(real_multiplier)
    if shift < 0:
        raise RuntimeError(
            f"Softmax input scale {input_scale} is below what esp-nn can represent."
        )
    max_input_rescaled = (
        ((1 << SOFTMAX_INPUT_INTEGER_BITS) - 1)
        * math.ldexp(1.0, 31 - SOFTMAX_INPUT_INTEGER_BITS)
        / math.ldexp(1.0, shift)
    )
    return multiplier, shift, -int(math.floor(max_input_rescaled))


@AtenToEspressifPass.register_dialect_substitution(exir_ops.edge.aten._softmax.default)
def _get_softmax_replacement(
    node: Node, dialect_pass: AtenToDialectPass
) -> DialectNodeSpec | None:
    del dialect_pass
    if not _has_qparams(node):
        return None

    tensor = get_first_fake_tensor(node)
    dim = cast(int, node.args[1]) % tensor.dim()
    if dim != tensor.dim() - 1:
        raise RuntimeError(f"{node.name}: esp-nn softmax only reduces the last dim.")
    output_zp, _, _, output_scale = _output_range(node)
    if output_zp != SOFTMAX_OUTPUT_ZERO_POINT or not math.isclose(
        output_scale, SOFTMAX_OUTPUT_SCALE, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"{node.name}: softmax output must be quantized with scale 1/256 and "
            "zero point -128."
        )

    multiplier, shift, diff_min = _softmax_parameters(
        float(node.meta["input_qparams"][0].scale)
    )
    args = (node.args[0], multiplier, shift, diff_min)
    return DialectNodeSpec(exir_ops.edge.espressif.quantized_softmax.default, args)
