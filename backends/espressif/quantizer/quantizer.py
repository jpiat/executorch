# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PT2E quantizer producing the int8 scheme esp-nn executes.

esp-nn uses the TensorFlow Lite Micro int8 scheme, which is also the scheme
of CMSIS-NN, so the quantization specs and the pattern-quantizer machinery
are shared with the Cortex-M backend. Only the set of supported patterns
differs.

- activations: int8, per-tensor, asymmetric
- conv2d / linear weights: int8, per-output-channel, symmetric
- bias: int32 with scale input_scale * weight_scale
- softmax output: fixed scale 1/256, zero point -128
- pooling: output shares the input quantization
"""

from typing import cast, Optional

from executorch.backends.arm.quantizer.arm_quantizer_utils import (
    PatternCheck,
    PatternQuantizer,
    SharedQspecQuantizer,
)
from executorch.backends.cortex_m.quantizer.node_finders import (
    GlobalNoOutputNodeFinder,
    NodeTargetNodeFinder,
)
from executorch.backends.cortex_m.quantizer.pattern_matcher import PatternMatcher
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    INT8_PER_CHANNEL_CONFIG,
    INT8_PER_TENSOR_CONFIG,
)
from executorch.backends.cortex_m.quantizer_reporter import QuantizerReporter
from executorch.backends.espressif.quantizer.quantizer_support import (
    __name__ as espressif_quantizer_support_module,
    CONV_OP_PATTERNS,
    ESPRESSIF_QUANTIZER_SUPPORT_DICT,
    LINEAR_OP_PATTERNS,
)
from torch._ops import OpOverload
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantizer import ComposableQuantizer


class EspressifQuantizer(ComposableQuantizer):
    """Quantizer targeting the esp-nn kernels of the Espressif backend.

    Convolutions and linear layers get per-channel weights; every other
    supported pattern is quantized per-tensor. Data-movement ops between
    quantized ops share their input's quantization and run on the portable
    int8 kernels.
    """

    def __init__(self) -> None:
        pattern_matcher = PatternMatcher(
            cast(
                dict[tuple[OpOverload, ...], Optional[type[PatternCheck]]],
                ESPRESSIF_QUANTIZER_SUPPORT_DICT,
            ),
            support_dict_name=espressif_quantizer_support_module
            + ".ESPRESSIF_QUANTIZER_SUPPORT_DICT",
        )

        per_channel_targets: set[OpOverload] = set()
        for pattern in CONV_OP_PATTERNS.keys() | LINEAR_OP_PATTERNS.keys():
            per_channel_targets.update(pattern)

        super().__init__(
            [
                PatternQuantizer(
                    INT8_PER_CHANNEL_CONFIG,
                    node_finder=NodeTargetNodeFinder(list(per_channel_targets)),
                    pattern_matcher=pattern_matcher,
                ),
                PatternQuantizer(
                    INT8_PER_TENSOR_CONFIG,
                    node_finder=GlobalNoOutputNodeFinder(),
                    pattern_matcher=pattern_matcher,
                ),
                SharedQspecQuantizer(),
            ]
        )

    def annotate(self, model: GraphModule) -> GraphModule:
        reporter = QuantizerReporter(self.quantizers)
        model = super().annotate(model)
        reporter.log_quantizer_report(model)
        return model

    def validate(self, model: GraphModule) -> None:
        return None

    def transform_for_annotation(self, model: GraphModule) -> GraphModule:
        from executorch.backends.espressif.passes.espressif_pass_manager import (
            EspressifPassManager,
        )

        return EspressifPassManager.transform_for_annotation(model)
