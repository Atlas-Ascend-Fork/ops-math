# Sqrt

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |     √      |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |   ×     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: computes the square root of non-negative numbers, and returns nan for negative numbers.
- Computation formula:

  $$
  out=sqrt(self)=\begin{cases}
  \sqrt {self}, & self\ge 0 , \\
  nan, &  self\lt 0
  \end{cases}
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
      <td>Input parameter for the sqrt computation, which is self in the formula.</td>
      <td>COMPLEX64, COMPLEX128, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the sqrt computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sqrt](./examples/test_aclnn_sqrt.cpp) | Invokes the Sqrt operator through the [aclnnSqrt](docs/aclnnSqrt&aclnnInplaceSqrt_en.md) API. |
| Graph-mode invocation | [test_geir_sqrt](./examples/test_geir_sqrt.cpp) | Invokes the Sqrt operator by building a graph through the [operator IR](./op_graph/sqrt_proto.h). |
