# IsNan

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |      ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: determines which elements in a tensor are nan.
- Computation formula:

  $$
  out_i=(input_i != input_i)
  $$

- Examples:
  - If x=[9, 6, 3], the computation result is [False, False, False].
  - If x=[[-3.14, inf], [2.7183, nan]], the computation result is [[False, False], [False, True]].

## Parameter Description

<table style="table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
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
      <td>Input input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output out_i in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_is_nan](./examples/test_geir_is_nan.cpp)   | Invokes the IsNan operator by building a graph through the [operator IR](./op_graph/is_nan_proto.h). |
