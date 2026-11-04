# Fast Kernel Launch

English | [简体中文](./README.md)

## Overview

This document demonstrates how to develop custom NPU operators using Ascend C and the [PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html) capabilities.

**Core advantages:**

- **Single deliverable:** One file completes both operator development and PyTorch framework adaptation.

- **Efficient invocation:** The kernel function is launched using the `<<<>>>` syntax, and the process is simple and efficient.

## Prerequisites

- Refer to [Environment Deployment](../../docs/zh/install/quick_install.md) first to complete the basic environment setup
- gcc 9.4.0+
- python 3.8+
- torch>=2.6.0
- [TorchNPU](https://gitcode.com/Ascend/pytorch/releases) of the corresponding version

## Installation Steps

1. Go to the `examples/fast_kernel_launch_example` directory.

2. Install Dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. Build the Wheel:

    ```sh
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

    By default, `dav-2201` is used as the NPU compilation architecture. You can specify a supported architecture through the `NPU_ARCH` environment variable:

    ```sh
    NPU_ARCH=dav-2201 python3 -m build --wheel -n
    ```

    The currently supported values of `NPU_ARCH` are `dav-2201` and `dav-3510`.

    After the build is complete, the artifacts are in the `dist` folder in the current directory. The artifact name is `ascend_ops-1.0.0-${python_version}-abi3-${arch}.whl`,
    where `${python_version}` indicates the python version in the current environment (python3.8.3 corresponds to cp38), and `${arch}` indicates the CPU architecture.

4. Install the Wheel Package:

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

5. (Optional) Before building again, it is recommended that you run the following command to clear the compilation cache

   ```sh
    python setup.py clean
    ```

## Quick Start

After the installation is complete, you can use NPU operators in the same way as ordinary PyTorch operations. The invocation of the add operator is used as an example.

```python
import torch
import torch_npu
import ascend_ops  # Python package built

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)  # PyTorch Custom Operator Dispatch mechanism: torch.ops.<library_name>.<operator_name>

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

To implement a new operator (such as `add`), you only need to provide an `.asc` source file implementation.

1. First, create a folder named after the operator name `add` in the `csrc` directory, and create a subfolder named after the target `NPU_ARCH` (for example, `dav-2201`) in this folder.

2. Create a `CMakeLists.txt` file in the `NPU_ARCH` directory

    ```cmake
    ascend_ops_add_current_op(OP_TARGET)
    ```

    Here, `NPU_ARCH` is passed to the `--npu-arch` parameter of the compiler. Currently, `dav-2201` and `dav-3510` are supported. For how to obtain the value, refer to [NpuArch Description and Usage Guide](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md). If the operator requires additional dependencies, you can explicitly add them for `${OP_TARGET}` in the same `CMakeLists.txt`.

3. Create an `add.asc` file in the `NPU_ARCH` directory (it is recommended that you use the operator name as the file name). This file contains all modules required for developing an AI Core operator.
    - Operator schema registration
    - Operator Meta Function implementation and registration
    - Operator kernel implementation (Ascend C)
    - Operator NPU invocation implementation and registration

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // The current project is a namespace
    namespace Add {         // It is recommended that each operator have its own independent namespace to prevent global variable pollution

    /**
     * Registers the operator schema with the PyTorch framework
     * The framework knows that this operator exists
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * Implements the Meta function of the operator, that is, InferShape+InferDtype
     * Deduces what the output of this operator looks like and how much space is needed based on the input, without actually computing this operator
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * Registers the Meta function of the operator with the framework
     * The framework can call this Meta function to know how much space is needed before actually executing the operator computation
     * Later, graph acceleration such as torch.compile/AutoGrad/AclGraph can be supported
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * Implements the kernel of the NPU operator using the AscendC API, written for the current soc
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * Implements the operator invocation API
     * In this API, the invocation of the NPU kernel needs to be completed
     * 1. Compute the number/shape/dtype of the output tensor (can be implemented by calling the Meta function or implemented directly)
     * 2. Compute the tiling: determine how to split the computation based on the shape
     * 3. Invoke the NPU kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        // OptionalDeviceGuard ensures that subsequent operations are executed in the correct device context
        // It records the current device state and automatically restores it after the scoped code is executed
        const c10::OptionalDeviceGuard guard(x.device());
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, numBlocks, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // Calls different NPU kernels based on different data types
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // The RunOpApi/RunOpApiV2 APIs must be used for the invocation to ensure that the timing is consistent with that when TorchNPU invokes the aclnn API.
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * Registers the invocation function of the operator with the framework, where Device is PrivateUse1
     * The framework knows that when the inputs are all on the NPU device, the dispatch goes to this operator implementation
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```

4. Rebuild and install the wheel package by referring to the [Installation Steps](#installation-steps) section.
5. For testing the operator API based on pytest, refer to the implementation of [test_add.py](tests/add/test_add.py).
