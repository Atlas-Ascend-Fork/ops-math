# End

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

Marks the output of a subgraph. The End operator receives the input tensor x, and the shape and data type of the output y are consistent with those of the input x.

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
      <td>peerIndex</td>
      <td>Attribute</td>
      <td>Index of the corresponding connected "placeholder" node.</td>
      <td>Int, with the default value 0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>parentOpType</td>
      <td>Attribute</td>
      <td>Operator type of the original node.</td>
      <td>String, with the default value ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Marked subgraph output tensor, whose shape and data type are consistent with those of the input x.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- End is an internal node. The system automatically inserts it at subgraph boundaries during graph compilation. It must be used together with the paired PlaceHolder node and the peerIndex attribute, and cannot be compiled and executed independently in user-built graphs.

## Invocation Description

The End operator is an internal basic operator. The system automatically inserts it during subgraph splitting, and it is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_end](./examples/test_geir_end.cpp) | Invokes the End operator by building a graph through the [operator IR](./op_graph/end_proto.h) |
