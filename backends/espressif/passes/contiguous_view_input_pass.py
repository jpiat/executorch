# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassResult

_CONTIGUOUS_ONLY = {
    exir_ops.edge.aten.view_copy.default,
}


class ContiguousViewInputPass(ExportPass):
    """Copy channels-last tensors to contiguous memory before reshaping them.

    A view reinterprets its input's memory in contiguous order. When a model
    is quantized, the graph is re-exported through quantize/dequantize nodes
    that report contiguous strides, so to_edge sees no layout change in front
    of e.g. a flatten and inserts no copy. The spatial Espressif operators do
    produce channels-last tensors, so the copy is restored here; without it
    the view would alias NHWC bytes as NCHW.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target not in _CONTIGUOUS_ONLY:
                continue
            source = node.args[0]
            value = source.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.is_contiguous():
                continue

            with graph_module.graph.inserting_before(node):
                copy = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.dim_order_ops._clone_dim_order.default,
                    args=(source,),
                    kwargs={"dim_order": list(range(value.dim()))},
                )
            copy.meta = dict(source.meta)
            copy.meta["val"] = value.contiguous()
            node.replace_input_with(source, copy)
            modified = True

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
