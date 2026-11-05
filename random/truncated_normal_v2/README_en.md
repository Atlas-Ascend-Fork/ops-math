# TruncatedNormalV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                         |    √     |
| Atlas A3 products     |    ×     |
| Atlas A2 products |    ×     |
| Atlas 200I/500 A2 inference products                      |    ×     |
| Atlas inference products                              |    ×     |
| Atlas training products                              |    ×     |

## Function Description

- Operator function: returns a sequence of random numbers that follows the standard normal distribution.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
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
      <td>shape</td>
      <td>Input</td>
      <td>Shape of the output tensor.</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset value.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Attribute</td>
      <td>Random number seed.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed2</td>
      <td>Attribute</td>
      <td>Random number seed.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Specifies the data type of the output.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Generated random number sequence.</td>
      <td>FLOAT16, FLOAT, BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_truncated_normal_v2](./examples/test_geir_truncated_normal_v2.cpp)   | Invokes the TruncatedNormalV2 operator by building a graph through the [operator IR](./op_graph/truncated_normal_v2_proto.h). |
