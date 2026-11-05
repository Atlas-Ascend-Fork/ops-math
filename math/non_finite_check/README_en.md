# NonFiniteCheck

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: checks whether non-finite numbers (+inf, -inf, NaN) exist in the list `tensor_list` that contains multiple tensors. If a non-finite number exists, found_flag is set to 1. Otherwise, found_flag is set to 0.

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
      <td>tensor_list</td>
      <td>Input</td>
      <td>List of tensors to be checked</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_flag</td>
      <td>Output</td>
      <td>Indicates whether non-finite numbers exist in `tensor_list`</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_non_finite_check](./examples/test_geir_non_finite_check.cpp)   | Invokes the NonFiniteCheck operator by building a graph through the [operator IR](./op_graph/non_finite_check_proto.h). |
