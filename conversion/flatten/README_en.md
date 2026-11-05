# Flatten

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- **Operator function**: flattens the input Tensor into a 2D Tensor based on the given axis.
- **Computation formula**:
  - If the shape of the input self is (d₀, d₁, ..., dₙ), the shape of the output out is (d₀×d₁×...×dₐₓᵢₛ₋₁, dₐₓᵢₛ×dₐₓᵢₛ₊₁×...×dₙ)
  - If axis is 0, the shape of the output out is (1, d₀×d₁×...×dₙ)

## Parameter Description

<table style="table-layout: fixed; width: 949px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 335px">
<col style="width: 102px">
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
    <td>Input tensor</td>
    <td>Input tensor. Non-contiguous Tensors are supported, 2D to 8D.</td>
    <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, BFLOAT16, FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>axis</td>
    <td>Input attribute</td>
    <td>Reference axis for the flatten computation. The value range is [-self.dim(), self.dim()-1]. Currently only axis = 1 is supported.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>Output 2D tensor. Non-contiguous Tensors are supported.</td>
    <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, BFLOAT16, FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- The number of dimensions of the input tensor cannot be greater than 8.
- axis must be within the valid dimension range.
- The input and output must have the same data type.

## Invocation Description

| Invocation Method  | Sample Code                                                | Description                                                         |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_flatten](./examples/test_aclnn_flatten.cpp) | Invokes the Flatten operator through the [aclnnFlatten](docs/aclnnFlatten_en.md) API. |
