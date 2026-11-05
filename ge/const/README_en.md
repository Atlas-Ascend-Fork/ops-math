# Const

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

Creates a constant tensor from a tensor-like object for inference scenarios. The Const operator is defined in the same way as the Constant operator. It infers the shape and data type of the output tensor y based on the Tensor value and type stored in the value attribute. The output shape and data type are consistent with the Tensor in the value attribute.

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
      <td>value</td>
      <td>Attribute</td>
      <td>Required. Value and type of the output tensor. The type is unlimited.</td>
      <td>Tensor, with the default value Tensor()</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Constant tensor, whose shape and data type are consistent with those of the value attribute.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The Const operator is a basic operator of the graph engine. It is usually used as a constant node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_const](./examples/test_geir_const.cpp) | Invokes the Const operator by building a graph through the [operator IR](./op_graph/const_proto.h) |
