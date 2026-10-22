# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.espressif.ops.operators  # noqa: F401
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ReplaceQuantNodesPass(ExportPass):
    """Run the graph's int8 per-tensor (de)quantize nodes on the Espressif
    kernels, so the firmware does not need the quantized kernel library."""

    _REPLACEMENTS = {
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: (
            exir_ops.edge.espressif.quantize_per_tensor.default
        ),
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: (
            exir_ops.edge.espressif.dequantize_per_tensor.default
        ),
    }

    @staticmethod
    def _is_int8(args) -> bool:
        return (
            args[5] == torch.int8
            and args[3] >= torch.iinfo(torch.int8).min
            and args[4] <= torch.iinfo(torch.int8).max
        )

    def call_operator(self, op, args, kwargs, meta):
        replacement = self._REPLACEMENTS.get(op)
        if replacement is not None and self._is_int8(args):
            op = replacement
        return super().call_operator(op, args, kwargs, meta)
