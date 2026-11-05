# VarIsInitializedOp

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

Checks whether a tensor (variable) has been initialized. The output is a bool scalar that indicates whether the input variable has been initialized: the input must be a Variable node. The graph engine queries the initialization state of the Variable through VarIsInitializedOpPass during compilation and rewrites this operator node into a constant output.

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
      <td>The variable tensor to check (which must be the output of a Variable node).</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>A bool scalar that indicates whether the variable has been initialized (regardless of the input shape/dtype).</td>
      <td>TensorType({DT_BOOL})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- This operator is an experimental feature (EXPERIMENTAL). Do not use it in the production environment.
- The input must be a Variable node: the graph engine VarIsInitializedOpPass queries the initialization state of the Variable during compilation and rewrites this operator node.
- The output is always a bool scalar, regardless of the shape and data type of the input.

## Invocation Description

The VarIsInitializedOp operator is a basic graph engine operator, and usually works with Variable in the mode of `Variable → VarIsInitializedOp` to query the initialization state of a variable.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_var_is_initialized_op](./examples/test_geir_var_is_initialized_op.cpp) | Invokes the VarIsInitializedOp operator by building a graph through the [operator IR](./op_graph/var_is_initialized_op_proto.h). |
