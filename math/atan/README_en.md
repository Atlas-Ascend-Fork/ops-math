# Atan

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: performs the arctangent operation on each element of the input and outputs the result.

- Computation formula:

  $$
  y_{i}=tan^{-1}(x_{i})
  $$

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor x in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>: Supports FLOAT, FLOAT16, and BFLOAT16.
- <term>Atlas inference products</term> and <term>Atlas training products</term>: BFLOAT16 is not supported.

## Constraint Description

- When the input type is INT8, INT16, INT32, INT64, UINT8, or BOOL, the input is converted to FLOAT32 for computation, and the output is of the FLOAT32 type. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the supported [data format](../../docs/en/context/data_format.md) is ND. The dimensions of a non-contiguous tensor must not be greater than 8, and the shapes of the input and output must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_Atan](./examples/test_aclnn_atan.cpp) | Invokes the Atan operator through the [aclnnAtan](./docs/aclnnAtan&aclnnInplaceAtan_en.md) API.  |
| Graph-mode invocation | [test_geir_Atan](./examples/test_geir_atan.cpp)   | Invokes the Atan operator by building a graph through the [operator IR](./op_graph/atan_proto.h).                                      |
