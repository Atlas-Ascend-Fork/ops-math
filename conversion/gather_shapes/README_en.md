# GatherShapes

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

- Operator function: collects the specified dimension values from the shapes of multiple input tensors and concatenates them into a one-dimensional tensor output in the order of axes.

- Computation logic:
  - Each element in axes is a two-element tuple [input_index, dim_index], which indicates taking the dim_index-th dimension value of the input_index-th input
  - The output is a one-dimensional tensor whose length equals the number of elements of axes and whose content is the dimension values collected by axes

- Example:

  ```text
  Input: x0 = float32[2, 3], x1 = float32[4, 5, 6], axes = [[0,0],[0,1],[1,2]]
  Output: shape = int32[3] = [2, 3, 6]
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
      <td>List of input tensors whose shape information is to be collected.</td>
      <td>Any data type</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Output</td>
      <td>One-dimensional tensor composed of the input dimension values collected by axes. Its length equals the number of elements of axes.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Attribute</td>
      <td>Required. List of two-element tuples. Each element [input_index, dim_index] specifies collecting the dim_index-th dimension value of the input_index-th input.</td>
      <td>ListListInt (list of int64 two-element tuples)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Data type of the output tensor. By default, INT32 is used when it is omitted.</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- axes cannot be empty, and each element must be a two-element tuple. input_index and dim_index cannot be out of bounds (this check is skipped when there are inputs with unknown rank).
- The dtype attribute supports only INT32 and INT64. Operator inference fails for other values.
- Only the ND format is supported for inputs.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_gather_shapes](./examples/test_geir_gather_shapes.cpp)   | Invokes the GatherShapes operator by building a graph through the [operator IR](./op_graph/gather_shapes_proto.h).                   |
