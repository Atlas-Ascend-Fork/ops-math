# RefData

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

Provides input data that can be overwritten by references for other operators. The RefData operator serves as a data input node in a graph. It directly passes the input tensor x to the output tensor y, and identifies the sequence number of the data node in the network through the index attribute. Different from the Data operator, RefData can be overwritten by ref-type operators and serves as a variable.

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
      <td>Index of the input tensor. The data type is int32 or int64. Assume that a network has two data nodes and one ref_data node: the indexes of the first two data nodes are set to 0 and 1, and the index of ref_data should be set to 2.</td>
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

The RefData operator is a basic operator of the graph engine. It is usually used as a data input node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_ref_data](./examples/test_geir_ref_data.cpp) | Invokes the RefData operator by building a graph through the [operator IR](./op_graph/ref_data_proto.h) |
