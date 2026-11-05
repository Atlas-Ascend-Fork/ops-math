# Rsqrt

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the reciprocal of the square root of each element in the input tensor x, and stores the result in the output tensor y.

- Computation formula:

$$
y = \frac{1}{\sqrt{x}}
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which is y in the formula. Its shape is the same as that of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 200I/500 A2 inference products, Atlas inference products, and Atlas training products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_rsqrt](./examples/test_aclnn_rsqrt.cpp) | Invokes the Rsqrt operator through the [aclnnRsqrt or aclnnInplaceRsqrt](./docs/aclnnRsqrt&aclnnInplaceRsqrt_en.md) API. |
| Graph-mode invocation | [test_geir_rsqrt](./examples/test_geir_rsqrt.cpp)   | Invokes the Rsqrt operator by building a graph through the [operator IR](./op_graph/rsqrt_proto.h). |
