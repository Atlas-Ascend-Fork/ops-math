# Overview

English | [简体中文](./README.md)

> Note:
>
> - Currently, most operators in the operator library run on the AI Core, and a few operators run on the AI CPU. By default, operators mentioned in this project refer to AI Core operators.
> - For the introduction to AI Core and AI CPU, see the "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" section in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples of AI Core operators and AI CPU operators. Developers can refer to the corresponding implementations based on actual conditions.

## Directory Description

```text
├── examples
│   ├── add_example                # AI Core operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Keep the original file.
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph building
│   │   ├── op_host                # Implementations related to the operator information library, tiling, and InferShape
│   │   └── op_kernel              # Operator kernel directory
│   ├── add_example_c_api          # AI Core C API operator name (kernel developed using the C API/asc_simd)
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Keep the original file.
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph building
│   │   ├── op_host                # Implementations related to the operator information library, tiling, and InferShape
│   │   └── op_kernel              # Operator kernel directory (C API style)
│   ├── add_example_aicpu          # AI CPU operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file. Keep the original file.
│   │   ├── examples               # Operator usage examples
│   │   ├── op_graph               # Directory related to operator graph building
│   │   ├── op_host                # Implementations related to the operator information library and InferShape
│   │   └── op_kernel_aicpu        # Operator kernel directory
│   ├── fast_kernel_launch_example # Sample of fast end-to-end operator development in the PyTorch scenario
│   ├── CMakeLists.txt             # Operator compilation configuration file. Keep the original file.
│   └── README.md                  # Operator description document

```

## Operator Development Samples

|Sample Directory|Sample Description|Operator Development|Operator Invocation|
|---|------------------|---|---|
| add_example |Operator that implements the addition of two tensors.| For the end-to-end operator development process, see [AI Core Operator Development Guide](../docs/zh/develop/aicore_develop_guide.md). |For the invocation, see [README](add_example/README_en.md).|
 |add_example_c_api|Operator that implements the addition of two tensors. The kernel is developed in the C API (asc_simd) mode.|For the end-to-end operator development process, see [AI Core Operator Development Guide](../docs/zh/develop/aicore_develop_guide.md).|For the invocation, see [README](add_example_c_api/README_en.md).|
 |add_example_aicpu|Operator that implements the addition of two tensors.|For the end-to-end operator development process, see [AI CPU Operator Development Guide](../docs/zh/develop/aicpu_develop_guide.md).| For the invocation, see [README](add_example_aicpu/README_en.md). |
|fast_kernel_launch_example|Sample of fast end-to-end operator development in the PyTorch scenario.|For the end-to-end operator development process, see [PyTorch Operator Quick Development Guide](./fast_kernel_launch_example/README_en.md).| For the invocation, see [README](fast_kernel_launch_example/README_en.md). |
