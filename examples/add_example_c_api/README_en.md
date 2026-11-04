# AddExampleCApi

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|Atlas A3 products|√|
|Atlas A2 products|√|

## Function Description

- Operator function: performs the addition computation. The kernel is developed in the C API (asc_simd) development mode.

- Computation formula:

$$
y = x1 + x2
$$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| x1 | Input | Input parameter for the addition computation, which is x1 in the formula. | FLOAT | ND |
| x2 | Input | Input parameter for the addition computation, which is x2 in the formula. | FLOAT | ND |
| y | Output | Output parameter for the addition computation, which is y in the formula. | FLOAT | ND |

## Constraint Description

The input and output support only 4 dimensions, and only the FLOAT data type is supported.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|----------|---------|------|
| aclnn invocation | [test_aclnn_add_example_c_api](./examples/test_aclnn_add_example_c_api.cpp) | See [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md) to complete operator compilation and verification. |

## Kernel Development Mode

The kernel in this example is developed in the C API (asc_simd) mode, which is different from the AscendC API in the C++ class mode. The core APIs include:

- `asc_init()` - Initialization
- `asc_copy_gm2ub_sync()` / `asc_copy_ub2gm_sync()` - Data transfer
- `asc_add_sync()` - Vector computation
