# AngleV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>          |      ×   |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×    |
| <term>Atlas inference products</term>                       |     ×    |
| <term>Atlas training products</term>                       |     √    |
| <term>Kirin X90 processor series products</term>                       |     √    |
| <term>Kirin 9030 processor series products</term>                       |     √    |

## Function Description

- Operator function: computes the element-wise angle (in radians) of the input tensor.

- Computation formula:

$$
output_i=angle(input_i)
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
      <td>Input parameter for the angle computation, which is input<sub>i</sub> in the formula.</td>
      <td>FLOAT16, FLOAT, COMPLEX64, BOOL, UINT8, INT8, INT16, INT32, INT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the angle computation, which is output<sub>i</sub> in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: COMPLEX is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_angle_v2](./examples/test_aclnn_angle_v2.cpp) | Invokes the AngleV2 operator through the [aclnnAngleV2](./docs/aclnnAngleV2_en.md) API. |
| Graph-mode invocation | [test_geir_angle_v2](./examples/test_geir_angle_v2.cpp)   | Invokes the AngleV2 operator by building a graph through the [operator IR](./op_graph/angle_v2_proto.h). |
