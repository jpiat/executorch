# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Optional, Type

from executorch.backends.arm._passes import (
    DeduplicateGetAttrPass,
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.passes.activation_fusion_pass import (
    ActivationFusionPass,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
    LiftConstantTensorsPass,
)
from executorch.backends.cortex_m.passes.quantized_clamp_activation_pass import (
    QuantizedClampActivationPass,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
)
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.program._program import _transform
from torch.export import ExportedProgram
from torch.fx import GraphModule

from .aten_to_espressif_pass import AtenToEspressifPass
from .contiguous_view_input_pass import ContiguousViewInputPass
from .replace_quant_nodes_pass import ReplaceQuantNodesPass

PassClass = Type[ExportPass | ExportedProgramPassBase]


class _EspressifLoweringPass(ExportedProgramPassBase):
    """Run pass classes in order, constructing each with the current program
    when it asks for one."""

    def __init__(self, pass_classes: list[PassClass]) -> None:
        self.pass_classes = pass_classes

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        modified = False
        for pass_cls in self.pass_classes:
            kwargs: dict[str, Any] = {}
            if "exported_program" in inspect.signature(pass_cls).parameters:
                kwargs["exported_program"] = exported_program
            transformed = _transform(exported_program, pass_cls(**kwargs))
            modified |= transformed is not exported_program
            exported_program = transformed
        return ExportedProgramPassResult(exported_program, modified)


class EspressifPassManager(ExportedProgramPassManager):
    """Lower a PT2E-quantized edge program to esp-nn backed operators.

    Pass it to ``to_edge_transform_and_lower(..., transform_passes=...)`` or to
    ``EdgeProgramManager.transform``.
    """

    pass_list: list[PassClass] = [
        # Fold qparams onto max_pool2d itself rather than its getitem.
        RemoveGetItemPass,
        ReplaceScalarWithTensorArgPass,
        FoldAndAnnotateQParamsPass,
        # Fold relu/clamp/hardsigmoid into the producer's output range.
        ActivationFusionPass,
        # Run the remaining clamp-like activations on int8 portable kernels.
        QuantizedClampActivationPass,
        AtenToEspressifPass,
        # Spatial ops leave channels-last tensors that views would misread.
        ContiguousViewInputPass,
        ReplaceQuantNodesPass,
        LiftConstantTensorsPass,
        RemoveUnusedConstantsPass,
    ]

    pass_list_transform_for_annotation: list[Type[ExportPass]] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        DeduplicateGetAttrPass,
    ]

    def __init__(
        self,
        exported_program: ExportedProgram | None = None,
        passes: Optional[list[PassClass]] = None,
    ) -> None:
        self.exported_program = exported_program
        super().__init__(
            [_EspressifLoweringPass(passes if passes is not None else self.pass_list)]
        )

    @classmethod
    def transform_for_annotation(cls, model: GraphModule) -> GraphModule:
        for pass_cls in cls.pass_list_transform_for_annotation:
            model = pass_cls().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        if not isinstance(self.exported_program, ExportedProgram):
            raise ValueError(
                f"{type(self).__name__}.transform() needs an ExportedProgram."
            )
        result = self(self.exported_program)
        return result.exported_program if result.modified else self.exported_program
