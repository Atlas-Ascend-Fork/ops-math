# Pow

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>      |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- API function: completes the computation by using each element of exponent as the power of the corresponding element of input.

- Computation formula:

$$
out_i = input_i^{exponent_i}
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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Indicates the base. The data types of input and exponent satisfy the <a href ="../../docs/en/context/deduction_relationship.md">TensorScalar mutual deduction relationship</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>Input</td>
      <td>Indicates the exponent. The data types of exponent and input satisfy the <a href ="../../docs/en/context/deduction_relationship.md">TensorScalar mutual deduction relationship</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Indicates input raised to the power of exponent. The data type must be a data type to which the inferred data types of input and exponent can be converted (for details, see the <a href ="../../docs/en/context/conversion_relationship.md">mutual conversion relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

Operator constraint: for INT32 integer computation, a timeout occurs in scenarios beyond the following ranges.

| shape  | exponent_value|
|----|----|
|<=100000 (100 thousand) |-200000000~200000000 (200 million)|
|<=1000000 (1 million) |-20000000~20000000 (20 million)|
|<=10000000 (10 million) |-2000000~2000000 (2 million)|
|<=100000000 (100 million) |-200000~200000 (200 thousand)|
|<=1000000000 (1 billion) |-20000~20000 (20 thousand)|

## Invocation Description

| Invocation Method | Sample Code  | Description     |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_exp2](./examples/test_aclnn_exp2.cpp) | Invokes the Pow operator through the [aclnnExp2](./docs/aclnnExp2&aclnnInplaceExp2_en.md) API. |
| aclnn invocation | [test_aclnn_pow_tensor_tensor](./examples/test_aclnn_pow_tensor_tensor.cpp) | Invokes the Pow operator through the [aclnnPowTensorTensor](./docs/aclnnPowTensorTensor&aclnnInplacePowTensorTensor_en.md) API. |
| Graph-mode invocation | [test\_geir\_pow](./examples/test_geir_pow.cpp) | Invokes the Pow operator by building a graph through the [operator IR](./op_graph/pow_proto.h). |
