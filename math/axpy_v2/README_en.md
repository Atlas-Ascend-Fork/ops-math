# AxpyV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|----------------------------------------------|:----:|
| <term>Ascend 950PR&950DT products</term>       |  √   |
| <term>Atlas A3 products</term> |  √   |
| <term>Atlas A2 products</term> |  √   |
| <term>Atlas 200I/500 A2 inference products</term>          |  √   |
| <term>Atlas inference products</term>                    |  √   |
| <term>Atlas training products</term>                    |  √   |

## Function Description

- Operator function: multiplies each element in the source operand by the corresponding element in a tensor, and then adds the product to the corresponding element in the destination operand.

- Computation formula:

$out_i=self_i + alpha_i × other_i$

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
      <td>x1</td>
      <td>Input</td>
      <td>Input tensor self_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor other_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>Input tensor alpha_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor out_i in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32, UINT8, INT8, INT64, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_axpy_v2](./examples/test_geir_axpy_v2.cpp)   | Invokes the AxpyV2 operator by building a graph through the [operator IR](./op_graph/axpy_v2_proto.h) |
