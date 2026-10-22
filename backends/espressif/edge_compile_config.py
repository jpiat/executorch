# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import EdgeCompileConfig

# linear must survive to_edge: decomposed into addmm it would lose its qparams
# and stay on portable float kernels. hardsigmoid must stay whole so that
# ActivationFusionPass can fold it into the producer's quantization.
_PRESERVE_OPS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.hardsigmoid.default,
    torch.ops.aten.hardsigmoid_.default,
)


def espressif_edge_compile_config() -> EdgeCompileConfig:
    """The to_edge configuration the Espressif backend requires.

    Edge-dialect validation is off because quantized graphs carrying int8
    tensors through ATen ops do not pass it.
    """
    return EdgeCompileConfig(
        preserve_ops=list(_PRESERVE_OPS),
        _check_ir_validity=False,
    )
