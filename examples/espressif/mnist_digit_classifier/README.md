# MNIST Handwritten Digit Classifier for ESP32

> **:warning: <span style="color:red">**This example is not tested in CI. Use at your own risk.**</span>**

This example demonstrates running a small MLP neural network for MNIST
handwritten digit classification on ESP32/ESP32-S3 using the ExecuTorch
runtime. It is based on the
[Raspberry Pi Pico 2 MNIST example](../../raspberry_pi/pico2/) and adapted to
use the [Espressif executor runner](../executor_runner/).

The model is a 3-layer MLP (`TinyMLPMNIST`) with hand-crafted weights that
recognizes digits 0, 1, 4, and 7 using specific feature detectors (vertical
lines, horizontal lines, oval shapes, etc.).

## Prerequisites

1. **ESP-IDF v5.1+**: Install the ESP-IDF toolchain following the
   [official guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/).

2. **ExecuTorch**: Clone and set up ExecuTorch:
   ```bash
   pip install -e . --no-build-isolation
   ```

3. **Cross-compiled ExecuTorch libraries**: The build script handles this
   automatically using selective build (only operators needed by the model are
   included). You can also build manually—see the
   [main Espressif README](../README.md#cross-compiling-executorch).

## Quick Start

### 1. Export the MNIST model

```bash
python3 examples/espressif/mnist_digit_classifier/export_mlp_mnist.py
```

This creates `balanced_tiny_mlp_mnist.pte` (~5 KB).

### 2. Build and flash

The build script performs three steps:
1. Cross-compiles ExecuTorch for the ESP target with selective build
   (`-DEXECUTORCH_SELECT_OPS_MODEL=<pte>`) so only the operators used by the
   model are included.
2. Converts the `.pte` to a C header for embedding in firmware.
3. Builds the ESP-IDF project.

```bash
# Source ESP-IDF environment
. $IDF_PATH/export.sh

# Full build (cross-compile ExecuTorch + build firmware):
./examples/espressif/mnist_digit_classifier/build.sh \
    --target esp32s3 \
    --pte balanced_tiny_mlp_mnist.pte

# If ExecuTorch is already cross-compiled, skip that step:
./examples/espressif/mnist_digit_classifier/build.sh \
    --target esp32s3 \
    --pte balanced_tiny_mlp_mnist.pte \
    --skip-et-build

# Flash and monitor
cd examples/espressif/mnist_digit_classifier/project
idf.py -p /dev/ttyUSB0 flash monitor
```

### 3. Expected output

The firmware runs inference on four hardcoded ASCII art digit patterns (0, 1,
4, 7) and prints the neural network scores for each:

```
Starting MNIST digit classifier!
=== Digit 0 ===
  [ASCII art of digit 0]
Running inference...
  Digit 0: 12.345 <- PREDICTED
  Digit 1: -2.100
  ...
PREDICTED: 0 (Expected: 0) CORRECT!

=== Digit 1 ===
  ...
```

## How It Works

1. **Model export** (`export_mlp_mnist.py`): Creates a `TinyMLPMNIST` model
   with hand-crafted weights and exports it to `.pte` format using
   `torch.export` and `executorch.exir.to_edge`.

2. **PTE to header** (`pte_to_header.py`): Converts the `.pte` binary into a
   C header (`model_pte.h`) so the model data is compiled directly into the
   firmware.

3. **Runtime** (`main.cpp`): Uses the `et_runner_*` API from the espressif
   executor runner component to:
   - Initialize the ExecuTorch runtime and load the model
   - Convert ASCII art digit patterns to 28×28 float tensors
   - Run inference and display the classification results

## Project Structure

```
mnist_digit_classifier/
├── README.md                    # This file
├── build.sh                     # Build helper script
├── export_mlp_mnist.py          # Model export script
└── project/
    ├── CMakeLists.txt           # ESP-IDF project file
    ├── partitions.csv           # Flash partition table
    ├── sdkconfig.defaults       # Default ESP-IDF config
    ├── sdkconfig.defaults.esp32s3  # ESP32-S3 specific config
    └── main/
        ├── CMakeLists.txt       # Main component build file
        └── main.cpp             # MNIST digit classifier application
```
