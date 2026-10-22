# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an int8 model running on esp-nn through the Espressif backend.

The model is quantized with EspressifQuantizer and lowered with
EspressifPassManager. The resulting .bpte bundles test inputs with the
outputs of the lowered program evaluated on the host reference kernels, so a
device run checks the esp-nn kernels against them.

The portable operators the program still needs are written next to the
.bpte, ready for EXECUTORCH_SELECT_OPS_LIST.
"""

import argparse
import logging
from pathlib import Path

import torch
from executorch.backends.espressif.edge_compile_config import (
    espressif_edge_compile_config,
)
from executorch.backends.espressif.passes import EspressifPassManager
from executorch.backends.espressif.quantizer import EspressifQuantizer
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class Mlp(torch.nn.Module):
    """MNIST-sized multilayer perceptron."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Cnn(torch.nn.Module):
    """Small CNN covering every operator of the backend."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.dw = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.pw = torch.nn.Conv2d(16, 16, 1)
        self.pool = torch.nn.MaxPool2d(2)
        self.avg = torch.nn.AvgPool2d(2)
        self.fc = torch.nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        y = torch.relu(self.pw(torch.relu(self.dw(x))))
        x = self.avg(x + y)
        return torch.softmax(self.fc(torch.flatten(x, 1)), dim=-1)


MODELS = {"mlp": Mlp, "cnn": Cnn}


def portable_operators(program) -> list[str]:
    """Operators of the program not provided by the espressif kernels."""
    names = set()
    for plan in program.executorch_program.execution_plan:
        for op in plan.operators:
            if op.name.startswith("espressif::"):
                continue
            names.add(f"{op.name}.{op.overload}" if op.overload else op.name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="cnn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # The convolutions consume and produce channels-last tensors, so both the
    # module and its inputs are converted before export.
    model = MODELS[args.model]().eval().to(memory_format=torch.channels_last)

    def sample():
        return (torch.rand(1, 1, 28, 28).contiguous(memory_format=torch.channels_last),)

    calibration = [sample() for _ in range(args.calibration_samples)]
    example_inputs = calibration[0]

    captured = torch.export.export(model, example_inputs, strict=True).module()
    prepared = prepare_pt2e(captured, EspressifQuantizer())
    for inputs in calibration:
        prepared(*inputs)
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example_inputs, strict=True),
        compile_config=espressif_edge_compile_config(),
        transform_passes=EspressifPassManager(),
    )
    lowered = edge.exported_program().module()

    test_inputs = calibration[:2]
    test_cases = []
    for inputs in test_inputs:
        expected = lowered(*inputs)
        reference = model(*inputs)
        print(
            "max |int8 reference - fp32| = "
            f"{float((expected - reference).abs().max()):.4f}"
        )
        test_cases.append(MethodTestCase(inputs, expected))

    program = edge.to_executorch()
    bundle = BundledProgram(program, [MethodTestSuite("forward", test_cases)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))

    operators = portable_operators(program)
    ops_file = args.output.with_suffix(".ops")
    ops_file.write_text(",".join(operators) + "\n")
    print(f"Exported {args.output} ({len(program.buffer)} byte program)")
    print(f"Portable operators ({ops_file}): {','.join(operators)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
