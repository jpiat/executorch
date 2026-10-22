# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lowering tests for the Espressif operators.

Each test quantizes a small module with EspressifQuantizer, lowers it with
EspressifPassManager, checks the graph uses the expected esp-nn operators and
runs the lowered program on the reference implementations.
"""

import pytest
import torch
from executorch.backends.espressif.test.tester import (
    assert_close_to_eager,
    assert_lowered,
    assert_matches_fake_quant,
    assert_views_read_contiguous_inputs,
    espressif_op,
    lower,
    to_channels_last,
)

QUANTIZE = espressif_op("quantize_per_tensor")
DEQUANTIZE = espressif_op("dequantize_per_tensor")


def _calibration(make_inputs, count=8):
    return [make_inputs() for _ in range(count)]


def _run(module, make_inputs, expected, absent=(), eager_lsb=32.0, fq_lsb=1.0):
    torch.manual_seed(0)
    inputs = make_inputs()
    calibration = _calibration(make_inputs)
    lowered = lower(module, inputs, calibration)
    assert_lowered(lowered, expected, absent)
    # The reference kernels must reproduce PT2E's fake quantization up to
    # rounding; the looser float comparison only catches gross errors.
    for sample in calibration:
        assert_matches_fake_quant(lowered, sample, fq_lsb)
        assert_close_to_eager(lowered, sample, eager_lsb)
    assert_views_read_contiguous_inputs(lowered)
    # Serialization requires every operator to have an out variant.
    lowered.edge.to_executorch()
    return lowered


class Linear(torch.nn.Module):
    def __init__(self, activation=None, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=bias)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x) if self.activation else x


@pytest.mark.parametrize(
    "activation", [None, torch.nn.ReLU(), torch.nn.ReLU6(), torch.nn.Hardsigmoid()]
)
@pytest.mark.parametrize("bias", [True, False])
def test_linear(activation, bias):
    _run(
        Linear(activation, bias),
        lambda: (torch.randn(4, 32),),
        {espressif_op("quantized_linear"): 1, QUANTIZE: 1, DEQUANTIZE: 1},
        absent=["aten.linear.default", "aten.relu.default", "aten.hardtanh.default"],
    )


def test_linear_3d_input():
    _run(
        Linear(),
        lambda: (torch.randn(2, 3, 32),),
        {espressif_op("quantized_linear"): 1},
        absent=["aten.linear.default"],
    )


class Conv(torch.nn.Module):
    def __init__(self, groups=1, stride=1, padding=1, out_channels=16, relu=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            8, out_channels, 3, stride=stride, padding=padding, groups=groups
        )
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        return torch.relu(x) if self.relu else x


@pytest.mark.parametrize("stride,padding", [(1, 1), (2, 0), (2, 1)])
def test_conv2d(stride, padding):
    module = Conv(stride=stride, padding=padding).to(memory_format=torch.channels_last)
    _run(
        module,
        lambda: to_channels_last(torch.randn(1, 8, 12, 12)),
        {espressif_op("quantized_conv2d"): 1},
        absent=["aten.convolution.default", "aten.relu.default"],
    )


def test_conv2d_batched():
    module = Conv().to(memory_format=torch.channels_last)
    _run(
        module,
        lambda: to_channels_last(torch.randn(3, 8, 10, 10)),
        {espressif_op("quantized_conv2d"): 1},
        absent=["aten.convolution.default"],
    )


@pytest.mark.parametrize("out_channels", [8, 16])
def test_depthwise_conv2d(out_channels):
    module = Conv(groups=8, out_channels=out_channels).to(
        memory_format=torch.channels_last
    )
    _run(
        module,
        lambda: to_channels_last(torch.randn(1, 8, 12, 12)),
        {espressif_op("quantized_depthwise_conv2d"): 1},
        absent=["aten.convolution.default"],
    )


def test_conv2d_contiguous_stays_float():
    # Without channels-last activations the conv is not quantized at all,
    # rather than being lowered on a layout esp-nn cannot read.
    torch.manual_seed(0)
    inputs = (torch.randn(1, 8, 12, 12),)
    lowered = lower(Conv(), inputs)
    counts = lowered.op_counts()
    assert counts[espressif_op("quantized_conv2d")] == 0
    assert counts["aten.convolution.default"] == 1


def test_dilated_conv2d_stays_float():
    module = torch.nn.Conv2d(8, 8, 3, dilation=2).to(memory_format=torch.channels_last)
    inputs = to_channels_last(torch.randn(1, 8, 12, 12))
    counts = lower(module, inputs).op_counts()
    assert counts[espressif_op("quantized_conv2d")] == 0


class Add(torch.nn.Module):
    def __init__(self, relu=False):
        super().__init__()
        self.relu = relu

    def forward(self, x, y):
        z = x + y
        return torch.relu(z) if self.relu else z


@pytest.mark.parametrize("relu", [False, True])
def test_add(relu):
    _run(
        Add(relu),
        lambda: (torch.randn(2, 64), 3 * torch.randn(2, 64)),
        {espressif_op("quantized_add"): 1, QUANTIZE: 2},
        absent=["aten.add.Tensor", "aten.relu.default"],
    )


def test_add_broadcast_stays_float():
    inputs = (torch.randn(2, 64), torch.randn(1, 64))
    counts = lower(Add(), inputs).op_counts()
    assert counts[espressif_op("quantized_add")] == 0


class Mul(torch.nn.Module):
    def forward(self, x, y):
        return x * y


def test_mul():
    _run(
        Mul(),
        lambda: (torch.randn(2, 64), torch.randn(2, 64)),
        {espressif_op("quantized_mul"): 1},
        absent=["aten.mul.Tensor"],
    )


class Pool(torch.nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 1)
        self.pool = pool

    def forward(self, x):
        return self.pool(self.conv(x))


@pytest.mark.parametrize(
    "pool",
    [
        torch.nn.MaxPool2d(2),
        torch.nn.MaxPool2d(3, stride=2, padding=1),
        torch.nn.AvgPool2d(2),
        torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
    ],
)
def test_pool2d(pool):
    name = (
        "quantized_max_pool2d"
        if isinstance(pool, torch.nn.MaxPool2d)
        else "quantized_avg_pool2d"
    )
    module = Pool(pool).to(memory_format=torch.channels_last)
    _run(
        module,
        lambda: to_channels_last(torch.randn(1, 4, 9, 9)),
        {espressif_op(name): 1, espressif_op("quantized_conv2d"): 1},
    )


def test_avg_pool2d_include_pad_stays_float():
    module = Pool(torch.nn.AvgPool2d(3, stride=1, padding=1)).to(
        memory_format=torch.channels_last
    )
    inputs = to_channels_last(torch.randn(1, 4, 9, 9))
    counts = lower(module, inputs).op_counts()
    assert counts[espressif_op("quantized_avg_pool2d")] == 0


class Softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def test_softmax():
    _run(
        Softmax(),
        lambda: (torch.randn(4, 16),),
        {espressif_op("quantized_softmax"): 1, espressif_op("quantized_linear"): 1},
        absent=["aten._softmax.default"],
        fq_lsb=2.0,
    )


class SmallCnn(torch.nn.Module):
    """MNIST-sized CNN touching every operator family."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.dw = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw = torch.nn.Conv2d(8, 8, 1)
        self.pool = torch.nn.MaxPool2d(2)
        self.avg = torch.nn.AvgPool2d(2)
        self.fc = torch.nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        y = torch.relu(self.pw(torch.relu(self.dw(x))))
        x = self.avg(x + y)
        x = torch.flatten(x, 1)
        return torch.softmax(self.fc(x), dim=-1)


def test_small_cnn():
    module = SmallCnn().to(memory_format=torch.channels_last)
    _run(
        module,
        lambda: to_channels_last(torch.rand(1, 1, 28, 28)),
        {
            espressif_op("quantized_conv2d"): 2,
            espressif_op("quantized_depthwise_conv2d"): 1,
            espressif_op("quantized_max_pool2d"): 1,
            espressif_op("quantized_avg_pool2d"): 1,
            espressif_op("quantized_add"): 1,
            espressif_op("quantized_linear"): 1,
            espressif_op("quantized_softmax"): 1,
            QUANTIZE: 1,
            DEQUANTIZE: 1,
            # Restores NCHW order before the flatten.
            "dim_order_ops._clone_dim_order.default": 1,
        },
        absent=[
            "aten.convolution.default",
            "aten.linear.default",
            "aten.add.Tensor",
            "aten._softmax.default",
        ],
        fq_lsb=3.0,
    )
