# Data

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Provides input data for other operators. The Data operator serves as a data input node in a graph. It directly passes the input tensor x to the output tensor y, and identifies the sequence number of the data node in the network through the index attribute.

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
      <td>index</td>
      <td>Attribute</td>
      <td>Index of the input tensor. The data type is int32 or int64. Assume that a network has three data nodes: one should be set to 0, another to 1, and the third to 2.</td>
      <td>Int, with the default value 0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor that has the same shape and data type as the input tensor x</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The Data operator is a basic operator of the graph engine. It is usually used as a data input node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_data](./examples/test_geir_data.cpp) | Invokes the Data operator by building a graph through the [operator IR](./op_graph/data_proto.h) |
