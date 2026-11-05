# VarHandleOp

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

Creates a handle to a variable resource. The output tensor y is a scalar resource handle (DT_RESOURCE type). The shape and dtype attributes describe the shape and data type of the variable that the handle points to (rather than the shape of the output itself), for use by downstream resource operators (such as ReadVariableOp and AssignVariableOp). The container and shared_name attributes specify the container and shared name of the variable.

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
      <td>container</td>
      <td>Attribute</td>
      <td>Optional. The container where the variable is located.</td>
      <td>String, with the default value of ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shared_name</td>
      <td>Attribute</td>
      <td>Optional. The shared name of the variable.</td>
      <td>String, with the default value of ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Required. The data type of elements in the variable that the handle points to.</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Attribute</td>
      <td>Optional. The shape of the variable that the handle points to.</td>
      <td>ListInt, with the default value of UNKNOWN_SHAPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>A scalar resource handle (DT_RESOURCE type).</td>
      <td>TensorType({DT_RESOURCE})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- This operator is an experimental feature (EXPERIMENTAL). Do not use it in the production environment.
- This operator is a resource handle source node. Its output is a scalar DT_RESOURCE tensor, which has no runtime computation semantics. This operator cannot be executed independently through RunGraph/BuildModel.
- The shape/dtype attributes describe the shape and type of the variable that the handle points to (not the shape and type of the output itself). During inference, they are passed to downstream resource operators through the HandleShapesAndTypes mechanism of InferenceContext (framework peer propagation), and downstream operators read them through GetInputHandleShapesAndTypes.

## Invocation Description

The VarHandleOp operator is a basic graph engine operator. It serves as a resource handle source node and works with downstream resource operators (such as ReadVariableOp and AssignVariableOp). It is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_var_handle_op](./examples/test_geir_var_handle_op.cpp) | Invokes the VarHandleOp operator by building a graph through the [operator IR](./op_graph/var_handle_op_proto.h). |
