# TemporaryVariable

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

Creates a temporary variable tensor. After TemporaryVariable is used, the reference to the variable tensor must be passed to the paired DestroyTemporaryVariable operator for destruction. The shape and data type of the output tensor y are inferred based on the shape and dtype attributes: the output shape is the same as the shape attribute, and the output data type is the same as the dtype attribute.

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
      <td>shape</td>
      <td>Attribute</td>
      <td>Required. The shape of the variable tensor.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Required. The data type of elements in the variable tensor.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var_name</td>
      <td>Attribute</td>
      <td>Optional. The name of the variable to create.</td>
      <td>String, with the default value of ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The created variable tensor, whose shape is the same as the shape attribute and whose data type is the same as the dtype attribute.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- This operator is an experimental feature (EXPERIMENTAL). Do not use it in the production environment.
- This operator is a TensorFlow-compatible operator, and the graph engine registers it as a node (GeDeletedOp) to be eliminated in the graph optimization phase: in a real TensorFlow graph, the framework parser pairs it with DestroyTemporaryVariable and processes it in the parsing phase. Therefore, this operator cannot be compiled and executed independently through RunGraph/BuildModel.

## Invocation Description

The TemporaryVariable operator is a basic graph engine operator, and is used in pairs with [DestroyTemporaryVariable](../destroy_temporary_variable/README_en.md): this operator creates a temporary variable for downstream writing, and DestroyTemporaryVariable reads the final value and destroys the temporary variable after all uses are completed.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_temporary_variable](./examples/test_geir_temporary_variable.cpp) | Invokes the TemporaryVariable operator by building a graph through the [operator IR](./op_graph/temporary_variable_proto.h). |
