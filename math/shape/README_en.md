# Shape

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns the shape of the input tensor, that is, a 1-dimensional tensor composed of the dimension values of the input.

- Computation logic:
  - The rank of the input tensor is r
  - The output is a 1-dimensional tensor with the length of r, and its content is the dimension values of the input tensor

- Example:

  ```text
  Input: x = float32[2, 3, 4]
  Output: y = int32[3] = [2, 3, 4]
  ```

## Parameter Description

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 230px">
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
      <td>Input tensor whose shape information is to be obtained.</td>
      <td>Any data type</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>1-dimensional tensor composed of the dimension values of the input tensor. Its length equals the rank of the input.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Data type of the output tensor. If it is omitted, the default value is INT32.</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The dtype attribute supports only INT32 and INT64. The operator inference fails if another value is used.
- When the input has an unknown rank (-2), the output is 1-dimensional with an unknown shape (-1).
- Only the ND format is supported for the input.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_shape](./examples/test_geir_shape.cpp)   | Invokes the Shape operator by building a graph through the [operator IR](./op_graph/shape_proto.h).                   |
