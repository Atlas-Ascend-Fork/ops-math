# StatelessRandperm

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                      |    √     |
| Atlas A3 products     |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products                      |    √     |
| Atlas inference products                              |    √     |
| Atlas training products                              |    √     |

## Function Description

- Operator function: returns a random permutation of the integers from 0 to N-1.

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
      <td>n</td>
      <td>Input</td>
      <td>Number of generated results, which corresponds to N in the operator function.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Random number seed.</td>
      <td>INT64</td>
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
      <td>layout</td>
      <td>Attribute</td>
      <td>Storage format.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Specifies the data type of the output.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Element values after random permutation.</td>
      <td>INT64, INT32, INT16, UINT8, INT8, FLOAT16, FLOAT, DOUBLE, BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_randperm](examples/test_aclnn_randperm.cpp) | Invokes the StatelessRandperm operator through the [aclnnRandperm](docs/aclnnRandperm_en.md) API. |
