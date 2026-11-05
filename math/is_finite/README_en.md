# IsFinite

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |
| <term>Kirin X90 processor series products</term>                         |    √     |
| <term>Kirin 9030 processor series products</term>                   | √ |

## Function Description

- Operator function: determines which elements in a tensor are finite numbers, that is, not inf, -inf, or nan.
- Computation formula:
  $$
  y_i=(x_i != \pm inf) \: and \: (x_i!= nan)
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
      <td>Input tensor x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
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

- Atlas training products: BFLOAT16 is not supported.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_is_finite](./examples/test_aclnn_is_finite.cpp) | Invokes the IsFinite operator through the [aclnnIsFinite](./docs/aclnnIsFinite_en.md) API.    |
| Graph-mode invocation | [test_geir_is_finite](./examples/test_geir_is_finite.cpp)   | Invokes the IsFinite operator by building a graph through the [operator IR](./op_graph/is_finite_proto.h). |
