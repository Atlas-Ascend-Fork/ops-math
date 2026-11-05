# Round

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

## Function Description

* Rounds the values of the input tensor to the nearest integer. If the value is equidistant from two integers, rounds to the even integer.
* Rounds the elements of the input tensor to the specified number of decimal places.

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
      <td>Input tensor.</td>
      <td>BFLOAT16, FLOAT16,  FLOAT32,  DOUBLE,  INT32,  INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>decimals</td>
      <td>Attribute</td>
      <td>An optional integer attribute that specifies the number of decimal places to retain. The default value is 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Same as the input tensor x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

* When the input value is in [-0.5, -0], the output value is 0.
* For scenarios where decimals is not 0: if the input data exceeds the range (-347000, 347000), the precision may be affected.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_round](./examples/test_aclnn_round.cpp) | Invokes the Round operator through the [aclnnRound](./docs/aclnnRound&aclnnInplaceRound_en.md) API    |
| Graph-mode invocation | [test_geir_round](./examples/test_geir_round.cpp)   | Invokes the Round operator by building a graph through the [operator IR](./op_graph/round_proto.h). |
