# Abs

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: computes the absolute value of each element in the input tensor.

- Computation formula:

$$out=|self|$$

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
      <td>Input parameter for the abs computation, which is self in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the abs computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16, COMPLEX32, and COMPLEX64 are not supported.
- The UINT8 and BOOL types are supported only through the aclnn invocation.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_abs](./examples/test_aclnn_abs.cpp) | Invokes the Abs operator through the [aclnnAbs](./docs/aclnnAbs_en.md) API. |
| Graph-mode invocation | [test_geir_abs](./examples/test_geir_abs.cpp)   | Invokes the Abs operator by building a graph through the [operator IR](./op_graph/abs_proto.h). |
