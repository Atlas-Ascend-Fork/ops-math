# QueueData

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

Provides queue data for other operators. The QueueData operator serves as a queue data input node in a graph. It computes the length of the output tensor based on the output_types and output_shapes attributes (each item contains a 64-byte information header, dimension description, and data bytes), and outputs a one-dimensional tensor of the DT_UINT8 type.

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
      <td>index</td>
      <td>Attribute</td>
      <td>Index of the input tensor. The data type is int32 or int64. Assume that a network has three data nodes: one should be set to 0, another to 1, and the third to 2.</td>
      <td>Int, with the default value 0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queue_name</td>
      <td>Attribute</td>
      <td>Queue name.</td>
      <td>String, with the default value ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_types</td>
      <td>Attribute</td>
      <td>List of data types of the output data.</td>
      <td>ListType, with the default value {}</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_shapes</td>
      <td>Attribute</td>
      <td>List of shapes of the output data.</td>
      <td>ListListInt, with the default value {{}, {}}</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>One-dimensional tensor of the DT_UINT8 type, whose length is the total number of bytes computed based on output_types and output_shapes.</td>
      <td>DT_UINT8</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The lengths of output_types and output_shapes must be consistent.

## Invocation Description

The QueueData operator is a basic operator of the graph engine. It is usually used as a queue data input node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_queue_data](./examples/test_geir_queue_data.cpp) | Invokes the QueueData operator by building a graph through the [operator IR](./op_graph/queue_data_proto.h) |
