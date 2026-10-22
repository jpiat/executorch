# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers lowering small models through the Espressif pipeline in tests."""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from executorch.backends.espressif.edge_compile_config import (
    espressif_edge_compile_config,
)
from executorch.backends.espressif.passes import EspressifPassManager
from executorch.backends.espressif.quantizer import EspressifQuantizer
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@dataclass
class LoweredModel:
    eager: torch.nn.Module
    quantized: torch.fx.GraphModule
    edge: EdgeProgramManager

    def op_counts(self) -> Counter:
        graph = self.edge.exported_program().graph
        return Counter(
            node.target.__name__
            for node in graph.nodes
            if node.op == "call_function" and hasattr(node.target, "namespace")
        )

    def run(self, *inputs: torch.Tensor):
        return self.edge.exported_program().module()(*inputs)


def lower(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    calibration: Iterable[tuple[torch.Tensor, ...]] | None = None,
) -> LoweredModel:
    model = model.eval()
    captured = torch.export.export(model, example_inputs, strict=True).module()
    prepared = prepare_pt2e(captured, EspressifQuantizer())
    for sample in calibration or [example_inputs]:
        prepared(*sample)
    quantized = convert_pt2e(prepared)
    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example_inputs, strict=True),
        compile_config=espressif_edge_compile_config(),
        transform_passes=EspressifPassManager(),
    )
    return LoweredModel(model, quantized, edge)


def espressif_op(name: str) -> str:
    return f"espressif.{name}.default"


def assert_lowered(
    lowered: LoweredModel,
    expected: dict[str, int],
    absent: Sequence[str] = (),
) -> None:
    counts = lowered.op_counts()
    for op, count in expected.items():
        assert counts[op] == count, f"expected {count} x {op}, graph has {counts}"
    for op in absent:
        assert counts[op] == 0, f"{op} was not lowered: {counts}"


def assert_views_read_contiguous_inputs(lowered: LoweredModel) -> None:
    """On device a view aliases its input's bytes, so it must not read a
    channels-last tensor. The host reference cannot catch this: PyTorch's
    view_copy honours strides."""
    for node in lowered.edge.exported_program().graph.nodes:
        if node.op == "call_function" and node.target.__name__ == (
            "aten.view_copy.default"
        ):
            source = node.args[0].meta["val"]
            assert source.is_contiguous(), f"{node.name} reads a non-contiguous input"


def output_scales(lowered: LoweredModel) -> list[float]:
    """Quantization scale of each graph output, read from the dequantize node
    producing it."""
    graph = lowered.edge.exported_program().graph
    output_node = graph.output_node()
    scales = []
    for value in output_node.args[0]:
        assert (
            value.target.__name__ == "espressif.dequantize_per_tensor.default"
        ), f"output {value} is not produced by a dequantize node"
        scales.append(float(value.args[1]))
    return scales


def _compare(lowered, expected, actual, lsb: float, what: str) -> None:
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    for exp, act, scale in zip(expected, actual, output_scales(lowered)):
        error = float((exp - act).abs().max()) / scale
        assert (
            error <= lsb + 1e-3
        ), f"max error vs {what} is {error:.2f} LSB, allowed {lsb} LSB"


def assert_close_to_eager(
    lowered: LoweredModel, inputs: tuple[torch.Tensor, ...], lsb: float
) -> None:
    """Compare against the float model in units of the output scale. The
    inputs must lie within the calibrated range."""
    _compare(lowered, lowered.eager(*inputs), lowered.run(*inputs), lsb, "eager")


def assert_matches_fake_quant(
    lowered: LoweredModel, inputs: tuple[torch.Tensor, ...], lsb: float = 1.0
) -> None:
    """Compare against the PT2E fake-quantized model, which uses the same
    quantization parameters: each integer kernel may differ from it by one
    rounding step."""
    _compare(
        lowered, lowered.quantized(*inputs), lowered.run(*inputs), lsb, "fake-quant"
    )


def to_channels_last(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.contiguous(memory_format=torch.channels_last) for t in tensors)
