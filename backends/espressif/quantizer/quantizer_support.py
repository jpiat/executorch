# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator patterns the Espressif quantizer annotates.

Every pattern listed here must be lowered by AtenToEspressifPass; trailing
relu/hardtanh/clamp/hardsigmoid nodes are folded into the producer's output
range by ActivationFusionPass.
"""

import torch
from executorch.backends.espressif.quantizer.pattern_checkers import (
    EspressifAddMulCheck,
    EspressifAvgPool2DCheck,
    EspressifConv2DCheck,
    EspressifLinearCheck,
    EspressifMaxPool2DCheck,
    EspressifSoftmaxCheck,
)

aten = torch.ops.aten

_CLAMP_ACTIVATIONS = (
    aten.relu.default,
    aten.relu_.default,
    aten.hardtanh.default,
    aten.hardtanh_.default,
    aten.clamp.default,
    aten.clamp_.default,
)
_FUSABLE_ACTIVATIONS = _CLAMP_ACTIVATIONS + (
    aten.hardsigmoid.default,
    aten.hardsigmoid_.default,
)


def _with_activations(op, activations, check):
    patterns = {(op,): check}
    patterns.update({(op, activation): check for activation in activations})
    return patterns


LINEAR_OP_PATTERNS = _with_activations(
    aten.linear.default, _FUSABLE_ACTIVATIONS, EspressifLinearCheck
)

CONV_OP_PATTERNS = _with_activations(
    aten.conv2d.default, _FUSABLE_ACTIVATIONS, EspressifConv2DCheck
)

BINARY_OP_PATTERNS = {
    **_with_activations(aten.add.Tensor, _CLAMP_ACTIVATIONS, EspressifAddMulCheck),
    **_with_activations(aten.add_.Tensor, _CLAMP_ACTIVATIONS, EspressifAddMulCheck),
    (aten.mul.Tensor,): EspressifAddMulCheck,
    (aten.mul_.Tensor,): EspressifAddMulCheck,
}

POOL_OP_PATTERNS = {
    **_with_activations(
        aten.max_pool2d.default, _CLAMP_ACTIVATIONS[::2], EspressifMaxPool2DCheck
    ),
    (aten.avg_pool2d.default,): EspressifAvgPool2DCheck,
}

SOFTMAX_OP_PATTERNS = {
    (aten._softmax.default,): EspressifSoftmaxCheck,
    (aten.softmax.int,): EspressifSoftmaxCheck,
}

ESPRESSIF_QUANTIZER_SUPPORT_DICT = (
    LINEAR_OP_PATTERNS
    | CONV_OP_PATTERNS
    | BINARY_OP_PATTERNS
    | POOL_OP_PATTERNS
    | SOFTMAX_OP_PATTERNS
)
