# Acos

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

- Operator function: computes the inverse cosine of each element in the input and outputs the result.

- Computation formula:

  $$
  y_{i}=cos^{-1}(x_{i})
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
      <td>The input tensor x in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, BFLOAT16, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- When the input type is INT8, INT16, INT32, INT64, UINT8, or BOOL, the input is converted to FLOAT32 for computation, and the output is of the FLOAT32 type. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported. The supported [data format](../../docs/en/context/data_format.md) is ND. The number of dimensions of a non-contiguous tensor cannot be greater than 8, and the shapes of the input and output must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_acos](./examples/test_aclnn_acos.cpp) | Invokes the Acos operator through the [aclnnAcos](./docs/aclnnAcos&aclnnInplaceAcos_en.md) API.  |
| aclnn invocation | [test_aclnn_inplace_acos](./examples/test_aclnn_inplace_acos.cpp) | Invokes the Acos operator through the [aclnnInplaceAcos](./docs/aclnnAcos&aclnnInplaceAcos_en.md) API.  |
| Graph-mode invocation | [test_geir_acos](./examples/test_geir_acos.cpp)   | Invokes the Acos operator by building a graph through the [operator IR](./op_graph/acos_proto.h).                                      |
