# ReduceLogSumExp

English | [简体中文](./README.md)

## Product Support Status

| Product                                                                | Supported |
|:------------------------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>                            |    √     |
| <term>Atlas A3 products</term>                      |    √     |
| <term>Atlas A2 products</term>  |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                               |    ×     |
| <term>Atlas inference products</term>                                        |    √     |
| <term>Atlas training products</term>                                         |    √     |

## Function Description

- Operator function: returns the logarithm of the sum of exponentials of the input tensor along the specified dimensions.
- Computation formula:
  In the formula, i is the dimension specified by dim, and j is the element index of the input on the specified dimension.
  $$
  logsumexp(x)_i = log\sum_{j} exp(x_{ij} )
  $$

- Example

```text
Example 1:
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # dim_shape=[2] of the specified dimension, dim_data = {1, 2};
  keepDim: false
  out: [2]              # out_shape=[2];

Example 2:
  self: [2, 3, 4]       # self_shape=[2, 3, 4];
  dim: [2]              # dim_shape=[2] of the specified dimension, dim_data = {1, 2};
  keepDim: true
  out: [2, 1, 1]        # out_shape=[2, 1, 1];
```

## Parameter Description

- self (aclTensor*, computation input): `self` in the formula, an aclTensor on the Device side. The shape supports 0 to 8 dimensions, and the data type must be convertible to the out data type (see [Conversion Relationship](../../docs/en/context/conversion_relationship.md)). [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types supported are FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, and BOOL.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types supported are FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT16, INT8, UINT8, and BOOL.

- dim (aclIntArray*, computation input): dimensions participating in the computation, which is `i` in the formula, an aclIntArray on the Host side. The value range is [-self.dim(), self.dim()-1], and the supported data type is INT64.

- keepDim (bool, computation input): determines whether the dimensions of the reduce axis are retained, a BOOL constant on the HOST side.

- out (aclTensor*, computation input): $logsumexp(x)$ in the formula, an aclTensor on the Device side. The shape supports 0 to 8 dimensions. If keepDim is true, the shapes of dimensions other than the dimensions specified by dim, whose size is 1, must be the same as those of self. If keepDim is false, the dimensions of the reduce axis are not retained, and the shapes of the other dimensions must be the same as those of self. The data type must be convertible to the self data type (see [Conversion Relationship](../../docs/en/context/conversion_relationship.md)). [Non-contiguous Tensor](../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types supported are FLOAT and FLOAT16.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types supported are FLOAT, FLOAT16, and BFLOAT16.

- workspaceSize (uint64_t*, output parameter): returns the size of the workspace that needs to be allocated on the Device side.

- executor (aclOpExecutor**, output parameter): returns the op executor, which contains the operator computation process.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code                                                                         | Description                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_logsumexp.cpp](examples/test_aclnn_logsumexp.cpp) | Invokes the ReduceLogSumExp operator through the [aclnnLogSumExp](docs/aclnnLogSumExp_en.md) API. |
