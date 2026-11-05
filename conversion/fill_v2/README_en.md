# FillV2

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

- Operator function: creates an output tensor y filled with value according to the shape specified by dims.

- Computation formula:

$$y_i=value$$

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
      <td>dims</td>
      <td>Input</td>
      <td>1-D tensor that represents the shape of the output y. The number of dimensions does not exceed 8.</td>
      <td>INT16, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input attribute</td>
      <td>Value to be filled, which is value in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the computation, which is y_i in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input dims must be a 1-D tensor, and the number of dimensions does not exceed 8.
- The dimension values of the input dims cannot exceed the representation range of the data type of dims itself. If the range is exceeded, the values wrap around according to the type. If a value wraps to a negative number, the tiling verification intercepts it with an error, and the operator is not executed with the wrong shape. If the value is still non-negative after wrapping, the wrap cannot be detected, and the operator is executed with the wrapped value.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_fill_v2](./examples/test_geir_fill_v2.cpp)   | Invokes the FillV2 operator by building a graph through the [operator IR](./op_graph/fill_v2_proto.h). |
