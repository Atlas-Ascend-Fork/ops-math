# GetShape

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √   |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √   |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: obtains the shape information of one or more input tensors, and concatenates the size values of all dimensions in the input order into a one-dimensional int32 tensor output.

- Computation logic:
  - N tensors are input, and the rank of the i-th tensor is r_i
  - The output is a one-dimensional int32 tensor with a length of sum(r_i), and its content is the sequential concatenation of the dimension values of all input tensors

- Example:

  ```text
  Input: x0 = float32[2, 3, 4], x1 = float32[5, 6]
  Output: y = int32[5] = [2, 3, 4, 5, 6]
  ```

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
      <td>Dynamic input</td>
      <td>List of input tensors for obtaining shape information.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Sequential concatenation of the dimension values of all input tensors. The length is the sum of the ranks of all inputs.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The rank of each input tensor does not exceed 8.
- The sum of the ranks of all input tensors (that is, the output length) does not exceed 128 and is greater than 0 (all inputs cannot be scalars).
- The output dtype is fixed to INT32 and cannot be configured.
- Only the ND format is supported for inputs.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_get_shape](./examples/test_geir_get_shape.cpp)   | Invokes the GetShape operator by building a graph through the [operator IR](./op_graph/get_shape_proto.h).                   |
