#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export a small MLP model for MNIST digit classification to ExecuTorch .pte format.

This is the same model used in the Raspberry Pi Pico 2 MNIST example,
adapted for use with the Espressif ESP32 executor runner.

The model is a 3-layer MLP with hand-crafted weights that recognizes
digits 0, 1, 4, and 7 through specific feature detectors.

Usage:
    python3 export_mlp_mnist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir import EdgeCompileConfig, to_edge

from torch.export import export

INPUT_SIZE = 784  # 28*28 for MNIST
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 16
OUTPUT_SIZE = 10
IMAGE_SIZE = 28


class TinyMLPMNIST(nn.Module):
    """A small MLP for MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc3 = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_balanced_model():
    """
    Create a balanced MLP model for MNIST digit classification.

    The model is pre-initialized with specific weights to recognize
    digits 0, 1, 4, and 7 through hand-crafted feature detectors.
    """
    model = TinyMLPMNIST()

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.0)

        # Feature 0: Vertical lines (for 1, 4, 7)
        for row in range(IMAGE_SIZE):
            model.fc1.weight[0, row * IMAGE_SIZE + 14] = 2.0

        # Feature 1: Top horizontal (for 7, 4)
        model.fc1.weight[1, 0:84] = 2.0

        # Feature 2: Bottom horizontal (for 1, 4)
        model.fc1.weight[2, 25 * IMAGE_SIZE :] = 2.0

        # Feature 3: Oval detector for 0
        model.fc1.weight[3, 1 * IMAGE_SIZE + 8 : 1 * IMAGE_SIZE + 20] = 2.0
        model.fc1.weight[3, 26 * IMAGE_SIZE + 8 : 26 * IMAGE_SIZE + 20] = 2.0
        for row in range(4, 24):
            model.fc1.weight[3, row * IMAGE_SIZE + 7] = 2.0
            model.fc1.weight[3, row * IMAGE_SIZE + 20] = 2.0
        for row in range(10, 18):
            model.fc1.weight[3, row * IMAGE_SIZE + 14] = -1.5

        # Feature 4: Middle horizontal (for 4)
        model.fc1.weight[4, 13 * IMAGE_SIZE : 15 * IMAGE_SIZE] = 3.0

        # Second layer
        model.fc2.weight[0, 3] = 5.0
        model.fc2.weight[0, 0] = -2.0
        model.fc2.weight[0, 4] = -3.0

        model.fc2.weight[1, 0] = 3.0
        model.fc2.weight[1, 2] = 2.0
        model.fc2.weight[1, 1] = -1.0
        model.fc2.weight[1, 3] = -2.0

        model.fc2.weight[2, 0] = 2.0
        model.fc2.weight[2, 1] = 1.0
        model.fc2.weight[2, 4] = 4.0
        model.fc2.weight[2, 3] = -2.0

        model.fc2.weight[3, 1] = 3.0
        model.fc2.weight[3, 0] = 1.0
        model.fc2.weight[3, 2] = -2.0

        # Output layer
        model.fc3.weight[0, 0] = 5.0
        model.fc3.weight[1, 1] = 5.0
        model.fc3.weight[4, 2] = 5.0
        model.fc3.weight[7, 3] = 5.0

        for digit in [2, 3, 5, 6, 8, 9]:
            model.fc3.bias[digit] = -3.0

    return model


def main():
    print("Creating balanced MLP MNIST model...")

    model = create_balanced_model()

    example_input = torch.randn(1, IMAGE_SIZE, IMAGE_SIZE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    print("Exporting...")
    with torch.no_grad():
        exported_program = export(model, (example_input,))

    print("Converting to ExecuTorch...")
    edge_config = EdgeCompileConfig(_check_ir_validity=False)
    edge_manager = to_edge(exported_program, compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    filename = "balanced_tiny_mlp_mnist.pte"
    print(f"Saving {filename}...")
    with open(filename, "wb") as f:
        f.write(et_program.buffer)

    model_size_kb = len(et_program.buffer) / 1024
    print(f"Export complete! Model size: {model_size_kb:.1f} KB")


if __name__ == "__main__":
    main()
