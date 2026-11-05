# ReduceProd

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Operator function: returns the product of all elements in the input tensor.

## Parameter Description

  - self (aclTensor*, computation input): an aclTensor on the Device side. [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types supported are FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BOOL, COMPLEX64, and COMPLEX128. When the input is an empty tensor, the output type cannot be the complex types COMPLEX64 or COMPLEX128.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types supported are FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BOOL, COMPLEX64, COMPLEX128, and BFLOAT16. When the input is an empty tensor, the output type cannot be the complex types COMPLEX64 or COMPLEX128.

  - dtype (const aclDataType, computation input): an aclDataType on the Host side, the data type of the output tensor, which must be the same as the data type of out.

  - out (aclTensor*, computation output): an aclTensor on the Device side. The data type is the same as dtype, and if dtype is not passed in, the data type is the same as self. [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../docs/en/context/data_format.md) supports ND. The output shape is (1).
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types supported are FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BOOL, COMPLEX64, and COMPLEX128.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types supported are FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BOOL, COMPLEX64, COMPLEX128, and BFLOAT16.

## Constraint Description

- Deterministic computation:
  - Deterministic implementation by default.

## Invocation Example

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_prod](./examples/test_aclnn_prod.cpp) | Invokes the ReduceProd operator through the [aclnnProd](docs/aclnnProd_en.md) API. |
| aclnn API | [test_aclnn_prod_dim](./examples/test_aclnn_prod_dim.cpp) | Invokes the ReduceProd operator through the [aclnnProdDim](docs/aclnnProdDim_en.md) API. |
