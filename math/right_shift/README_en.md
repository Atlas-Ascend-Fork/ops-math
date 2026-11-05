# RightShift

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the bitwise right shift of the input tensors x and y element-wise.

- Computation formula:

$$
z_i = x_i >> y_i
$$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>The input x is a k-dimensional tensor. It is the input parameter for the right_shift computation, which is x_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Input parameter for the right_shift computation, which is y_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>negMode</td>
      <td>Attribute</td>
      <td>An optional integer attribute. The value is 0 or 1, and the default value is 0. If the value is 0 and y is negative, the computation is performed as a shift of 0 bits. If the value is 1 and y is negative, the computation is performed as a shift of the maximum number of bits.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Output parameter for the right_shift computation, which is z_i in the formula.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                          | Description                                                          |
|--------|---------------------------------------------------------------|-------------------------------------------------------------|
| aclnn invocation | [test_aclnn_rightshift](./examples/test_aclnn_right_shift.cpp) | Invokes the rightshift operator through the [aclnnRightshift](./docs/aclnnRightShift_en.md) API.|
| Graph-mode invocation | [test_geir_right_shift](./examples/test_geir_right_shift.cpp) | Invokes the RightShift operator by building a graph through the [operator IR](./op_graph/right_shift_proto.h). |
