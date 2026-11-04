# DestroyTemporaryVariable

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

Destroys a temporary variable and returns its final value. All other uses of the temporary variable must be executed before this operator. This operator is used in pairs with the TemporaryVariable operator: it matches TemporaryVariable through the var_name attribute, reads the final value of the temporary variable, and triggers the destruction. The shape and data type of the output tensor y are consistent with those of the input tensor x.

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
      <td>Reference to the temporary variable tensor.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_name</td>
      <td>Attribute</td>
      <td>Optional. Name of the temporary variable, which must be the same as the var_name attribute of the TemporaryVariable operator.</td>
      <td>String, with the default value ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final value of the temporary variable tensor, whose shape and data type are consistent with those of the input x.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- This operator is an experimental feature (EXPERIMENTAL) and must not be used in production environments.
- This operator is a TensorFlow-compatible operator. The graph engine registers it as a node that should be eliminated in the graph optimization phase (GeDeletedOp): in real TensorFlow graphs, the framework parser pairs it with TemporaryVariable and eliminates it during the parsing phase (after var_name matching, this node is deleted and consumers are directly connected). Therefore, this operator cannot be compiled and executed independently through RunGraph/BuildModel.

## Invocation Description

The DestroyTemporaryVariable operator is a basic operator of the graph engine and is used in pairs with the [TemporaryVariable](../temporary_variable/README_en.md) operator: TemporaryVariable creates a temporary variable for downstream writing, and this operator reads the final value and destroys the temporary variable after all uses are complete.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_destroy_temporary_variable](./examples/test_geir_destroy_temporary_variable.cpp) | Invokes the DestroyTemporaryVariable operator by building a graph through the [operator IR](./op_graph/destroy_temporary_variable_proto.h) |
