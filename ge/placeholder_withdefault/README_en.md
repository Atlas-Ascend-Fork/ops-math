# PlaceholderWithDefault

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

Inserts a placeholder with a default value for a tensor. The PlaceholderWithDefault operator receives the input tensor x, and the shape and data type of the output y are consistent with those of the input x. InferShape copies the input shape to the output, and InferDataType copies the input dtype to the output. The shape attribute specifies the shape information of the tensor.

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
      <td>shape</td>
      <td>Attribute</td>
      <td>Required. Shape of the tensor.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Created placeholder tensor, whose shape and data type are consistent with those of the input x.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The PlaceholderWithDefault operator is a basic operator of the graph engine. It is usually used as a placeholder node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_placeholder_withdefault](./examples/test_geir_placeholder_withdefault.cpp) | Invokes the PlaceholderWithDefault operator by building a graph through the [operator IR](./op_graph/placeholder_withdefault_proto.h) |
