# ReadVariableOp

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

Reads and returns the value of the input variable tensor. The shape and data type of the output tensor y are consistent with those of the input tensor x.

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
      <td>Variable tensor of a numeric type.</td>
      <td>TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE})</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Output data type, which is consistent with the input data type.</td>
      <td>Int, with the default value DT_INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor of a numeric type, whose shape and data type are consistent with those of the input x.</td>
      <td>TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The ReadVariableOp operator is a basic operator of the graph engine. It is usually used to read the value of a variable tensor. It is compatible with the TensorFlow operator ReadVariableOp.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_read_variable_op](./examples/test_geir_read_variable_op.cpp) | Invokes the ReadVariableOp operator by building a graph through the [operator IR](./op_graph/read_variable_op_proto.h) |
