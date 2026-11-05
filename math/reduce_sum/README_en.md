# ReduceSumOp

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

Operator function: returns the sum of each row of the input tensor over the given dimensions.

## Parameter Description

  - self (aclTensor*, compute input): aclTensor on the device. The shape supports 0-8 dimensions. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported. The supported [data format](../../docs/en/context/data_format.md) is ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, and COMPLEX128. When the input is an empty tensor, the output type cannot be the complex types COMPLEX64 and COMPLEX128.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, and BFLOAT16. When the input is an empty tensor, the output type cannot be the complex types COMPLEX64 and COMPLEX128.

  - dims (aclIntArray*, compute input): aclIntArray on the host. It specifies the reduce dimensions. The data type supports INT64, and the value range is [-self.dim(), self.dim()-1].

  - keepDims (bool, compute input): BOOL value on the host. It specifies whether to retain the dimensions of the input tensor in the output tensor.

  - dtype (aclDataType, compute input): aclDataType on the device. It specifies the data type of the output tensor.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT16, FLOAT32, INT8,  INT16,  INT32,  INT64, UINT8, BOOL, DOUBLE, COMPLEX64, and COMPLEX128.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT16, FLOAT32, INT8,  INT16,  INT32,  INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, and BFLOAT16.

  - out (aclTensor*, compute output): aclTensor on the device. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported. The supported [data format](../../docs/en/context/data_format.md) is ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types support FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, and COMPLEX128.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, BOOL, DOUBLE, COMPLEX64, COMPLEX128, and BFLOAT16.

## Constraint Description

- Deterministic computation:
  - Deterministic implementation by default.

## Invocation Example

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_reduce_sum](./examples/test_aclnn_reduce_sum.cpp) | Invokes the ReduceSum operator through the [aclnnReduceSum](docs/aclnnReduceSum_en.md) API. |
| Graph-mode invocation  | [test_geir_reduce_sum.cpp](./examples/test_geir_reduce_sum.cpp) | Invokes the ReduceSum operator through the GE IR graph mode. |
