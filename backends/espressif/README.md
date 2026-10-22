# Espressif Backend

> [!NOTE]
> Experimental. Validated bit-exact against the host reference on the ESP32-S3
> emulator with esp-nn's portable C kernels. The ESP32-S3 assembly kernels,
> which the emulator cannot run, still need validation on hardware.

## Overview

The Espressif backend runs int8 models on ESP32-S3 with
[esp-nn](https://github.com/espressif/esp-nn), Espressif's library of
optimized neural-network kernels. Like the Cortex-M backend, it is an
operator library rather than a delegate:

1. `EspressifQuantizer` quantizes the model with PT2E, using the scheme esp-nn
   implements.
2. `EspressifPassManager` replaces the quantized ATen operators with
   `espressif::` operators, each mapping to one esp-nn kernel.
3. The `espressif_kernels` ESP-IDF component implements those operators on
   esp-nn and registers them with the ExecuTorch runtime.

Anything the quantizer does not support stays in fp32 on the portable kernels.

## Quantization scheme

esp-nn follows the TensorFlow Lite Micro int8 scheme, which CMSIS-NN uses as
well, so the quantization specs come from the Cortex-M backend:

| Tensor | Quantization |
|--------|--------------|
| Activations | int8, per-tensor, asymmetric |
| Conv2d / linear weights | int8, per-output-channel, symmetric |
| Bias | int32, scale `input_scale * weight_scale` |
| Softmax output | fixed scale 1/256, zero point -128 |
| Pooling output | shares the input quantization |

Rescaling uses Q31 multipliers and power-of-two shifts computed ahead of time.

## Supported operators

| PyTorch | Espressif operator | esp-nn kernel | Constraints |
|---------|--------------------|---------------|-------------|
| `linear` | `quantized_linear` | `esp_nn_fully_connected_per_ch_s8` | |
| `conv2d`, groups = 1 | `quantized_conv2d` | `esp_nn_conv_s8` | channels-last, dilation 1 |
| `conv2d`, groups = in_channels | `quantized_depthwise_conv2d` | `esp_nn_depthwise_conv_s8` | channels-last, dilation 1 |
| `add` | `quantized_add` | `esp_nn_add_elementwise_s8` | same shapes, no broadcasting |
| `mul` | `quantized_mul` | `esp_nn_mul_elementwise_s8` | same shapes, no broadcasting |
| `max_pool2d` | `quantized_max_pool2d` | `esp_nn_max_pool_s8` | channels-last, no dilation or ceil mode |
| `avg_pool2d` | `quantized_avg_pool2d` | `esp_nn_avg_pool_s8` | channels-last; padding needs `count_include_pad=False` |
| `softmax` | `quantized_softmax` | `esp_nn_softmax_s8` | last dimension |
| quantize / dequantize | `quantize_per_tensor`, `dequantize_per_tensor` | | int8 |

A `relu`, `relu6`, `hardtanh`, `clamp` or `hardsigmoid` following linear or
conv2d, or a clamp-like activation following add or max_pool2d, is folded into
the producer's output range.

Spatial operators require channels-last activations: convert the module and
its inputs with `.to(memory_format=torch.channels_last)` before export.
Convolutions on contiguous tensors are not quantized and stay in fp32.

## Usage

```python
import torch
from executorch.backends.espressif.edge_compile_config import (
    espressif_edge_compile_config,
)
from executorch.backends.espressif.passes import EspressifPassManager
from executorch.backends.espressif.quantizer import EspressifQuantizer
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = model.eval().to(memory_format=torch.channels_last)
example_inputs = (x.contiguous(memory_format=torch.channels_last),)

prepared = prepare_pt2e(
    torch.export.export(model, example_inputs).module(), EspressifQuantizer()
)
for sample in calibration_data:
    prepared(*sample)
quantized = convert_pt2e(prepared)

program = to_edge_transform_and_lower(
    torch.export.export(quantized, example_inputs),
    compile_config=espressif_edge_compile_config(),
    transform_passes=EspressifPassManager(),
).to_executorch()
```

[`examples/espressif/export_quantized_model.py`](../../examples/espressif/export_quantized_model.py)
does this for an MLP and a small CNN and bundles reference outputs for an
on-device check. The [Espressif example README](../../examples/espressif/README.md#quantized-models-on-esp-nn)
covers the firmware build.

## Runtime integration

`espressif_kernels/` is an ESP-IDF component. It depends on
`espressif/esp-nn`, which the IDF component manager fetches. The kernels
register themselves from static initializers, so no operator YAML or code
generation is involved. To use them in an ESP-IDF project:

- add `backends/espressif/espressif_kernels` to `EXTRA_COMPONENT_DIRS` and
  require `espressif_kernels` from the component linking ExecuTorch (the
  example project does both with `-DET_ESPRESSIF_KERNELS=ON`);
- build the ExecuTorch runtime with `-DMAX_KERNEL_NUM` large enough for the
  selected portable kernels plus the ten espressif ones. Selective build
  otherwise sizes the registry for the portable kernels alone and the
  registration fails at boot;
- size the temp allocator pool (`ET_ESP_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE`)
  for the largest esp-nn scratch buffer of any single operator; the runtime
  frees temp memory after each instruction. The kernels request the size
  esp-nn reports and fail with `MemoryAllocationFailed` if it does not fit.

## Testing

```bash
pip install --no-dependencies -r backends/arm/requirements-arm-tosa.txt
pytest backends/espressif/test
```

The tests lower small models and run them on the Python reference operators,
which reproduce esp-nn's integer arithmetic; they must match PT2E's fake
quantization to within one rounding step. The quantizer and passes reuse Arm
and Cortex-M components, hence the TOSA requirement.

On-device numerics are checked by the emulator run in
[`.ci/scripts/test_espressif.sh`](../../.ci/scripts/test_espressif.sh), which
compares the firmware's outputs with the reference operators bit for bit.
