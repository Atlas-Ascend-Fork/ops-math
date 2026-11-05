# IsVariableInitialized

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

Checks whether a tensor is initialized. The IsVariableInitialized operator receives the input tensor x and outputs the scalar Boolean value y, indicating whether the input tensor is initialized. InferShape sets the output shape to a scalar (0-dimensional), and InferDataType sets the output data type to DT_BOOL.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>Input tensor.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Scalar Boolean tensor, indicating whether the input tensor is initialized. Its shape is a scalar (0-dimensional), and its data type is DT_BOOL.</td>
      <td>TensorType({DT_BOOL})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The output y is a scalar (0-dimensional tensor), and its data type is fixed to DT_BOOL.

## Invocation Description

The IsVariableInitialized operator is a basic operator of the graph engine. It is usually used to check whether a variable is initialized and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_is_variable_initialized](./examples/test_geir_is_variable_initialized.cpp) | Invokes the IsVariableInitialized operator by building a graph through the [operator IR](./op_graph/is_variable_initialized_proto.h) |
