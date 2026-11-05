# GreaterEqual

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: checks whether each element in the input tensor is greater than or equal to the value of the other scalar, and returns a tensor of the Bool type, indicating whether the greater-than-or-equal-to check at each position in the input tensor holds.

- Computation formula:
  For the input parameter self and the comparison scalar other, ge can be expressed by the following mathematical formula:

  $$
  y_{i}= (x1_i >= x2_i) ? True : False
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
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, UINT64, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor x2 in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, UINT64, INT8, UINT8, BOOL</td>
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
| aclnn API | [test_aclnn_ge_scalar](./examples/test_aclnn_ge_scalar.cpp) | Invokes the GreaterEqual operator through the [aclnnGeScalar&aclnnInplaceGeScalar](./docs/aclnnGeScalar&aclnnInplaceGeScalar_en.md) API.    |
| GE graph mode | [test_geir_greater_equal](./examples/test_geir_greater_equal.cpp) | Invokes the GreaterEqual operator by building a graph through the [operator IR](./op_graph/greater_equal_proto.h). |
