# Less

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: determines whether each element in the input self is less than the value of the input other, and returns a Tensor of the Bool type.

- Computation formula:

  $$
  y_i = (x1_i < x2_i)  ?  [True] : [False]
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
      <td>x1</td>
      <td>Input</td>
      <td>Input tensor x1 in the formula.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor x2 in the formula.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_lt_scalar](./examples/test_aclnn_lt_scalar.cpp) | Invokes the Less operator through the [aclnnLtScalar&aclnnInplaceLtScalar](./docs/aclnnLtScalar&aclnnInplaceLtScalar_en.md) API.    |
| GE graph mode | [test_geir_less](./examples/test_geir_less.cpp)   | Invokes the Less operator by building a graph through the [operator IR](./op_graph/less_proto.h). |
