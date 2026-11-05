# ShapeN

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

- Operator function: returns the shapes of multiple input tensors. Each input corresponds to one output, and the output is a 1-dimensional tensor composed of the dimension values of the corresponding input.

- Computation logic:
  - N tensors are input, and the rank of the i-th tensor is r_i
  - The output is N 1-dimensional tensors. The length of the i-th output is r_i, and its content is the dimension values of the i-th input

- Example:

  ```text
  Input: x0 = float32[2, 3], x1 = float32[4, 5, 6]
  Output: y0 = int32[2] = [2, 3], y1 = int32[3] = [4, 5, 6]
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
      <td>Dynamic input</td>
      <td>List of input tensors whose shape information is to be obtained.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, UINT8, UINT16, INT32, UINT32, INT64, UINT64, BOOL, STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Dynamic output</td>
      <td>Corresponds to the inputs one by one. Each output is a 1-dimensional tensor composed of the dimension values of the corresponding input, and its length equals the rank of the corresponding input.</td>
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
- When any input has an unknown rank (-2), the corresponding output is 1-dimensional with an unknown shape (-1).
- Only the ND format is supported for the input.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_shape_n](./examples/test_geir_shape_n.cpp)   | Invokes the ShapeN operator by building a graph through the [operator IR](./op_graph/shape_n_proto.h).                   |
