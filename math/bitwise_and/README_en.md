# BitwiseAnd

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

- Operator function: computes the bitwise AND of each element in the input tensor and the corresponding element of the other input (a tensor or a scalar). The inputs must be integer or BOOL variables. When the inputs are BOOL, the logical AND result is computed.
- Computation formula:

  - When other is a tensor:

    $$
    out_i = self_i \& other_i
    $$

  - When other is a scalar:

    $$
    out_i = self_i \& other
    $$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>INT16, UINT16, INT32, INT64, INT8, UINT8, UINT32, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td>INT16, UINT16, INT32, INT64, INT8, UINT8, UINT32, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>BOOL, INT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, UINT8, FLOAT, FLOAT16, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas A3 products, Atlas A2 products, Atlas inference products, and Atlas training products: the data types of self and other do not support UINT32 or UINT64; the data type of out does not support UINT32, UINT64, COMPLEX64, or COMPLEX128.

## Constraint Description

- The data types of the inputs must satisfy the data type deduction rules (see [deduction relationship](../../docs/en/context/deduction_relationship.md)), and the deduced data types must be within the supported range.
- Each API is deterministic by default. For specific constraints, refer to the corresponding aclnn API documentation.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_bitwise_and_scalar.cpp](examples/test_aclnn_bitwise_and_scalar.cpp) | Invokes the BitwiseAndScalar operator through the [aclnnBitwiseAndScalar](docs/aclnnBitwiseAndScalar_en.md) API. |
| aclnn API  | [test_aclnn_bitwise_and_tensor.cpp](examples/test_aclnn_bitwise_and_tensor.cpp) | Invokes the BitwiseAndTensor operator through the [aclnnBitwiseAndTensor](docs/aclnnBitwiseAndTensor_en.md) API. |
| aclnn API  | [test_aclnn_bitwise_and_tensor_out.cpp](examples/test_aclnn_bitwise_and_tensor_out.cpp) | Invokes the BitwiseAndTensorOut operator through the [aclnnBitwiseAndTensorOut](docs/aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut_en.md) API. |
| aclnn API  | [test_aclnn_inplace_bitwise_and_tensor_out.cpp](examples/test_aclnn_inplace_bitwise_and_tensor_out.cpp) | Invokes the InplaceBitwiseAndTensorOut operator through the [aclnnInplaceBitwiseAndTensorOut](docs/aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut_en.md) API. |
| aclnn API  | [test_aclnn_inplace_bitwise_and_scalar.cpp](examples/test_aclnn_inplace_bitwise_and_scalar.cpp) | Invokes the InplaceBitwiseAndScalar operator through the [aclnnInplaceBitwiseAndScalar](docs/aclnnInplaceBitwiseAndScalar_en.md) API. |
| aclnn API  | [test_aclnn_inplace_bitwise_and_tensor.cpp](examples/test_aclnn_inplace_bitwise_and_tensor.cpp) | Invokes the InplaceBitwiseAndTensor operator through the [aclnnInplaceBitwiseAndTensor](docs/aclnnInplaceBitwiseAndTensor_en.md) API. |
