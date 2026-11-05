# Fills

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: fills all elements of the input tensor with a specified value.

- Computation formula:

$$out_i=value$$

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
      <td>Input parameter for the Fills computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input attribute</td>
      <td>Fill value for the Fills computation, which is value in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the Fills computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_fills](./examples/test_geir_fills.cpp)   | Invokes the Fills operator by building a graph through the [operator IR](./op_graph/fills_proto.h). |
