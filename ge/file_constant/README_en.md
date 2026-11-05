# FileConstant

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

Creates a file constant tensor for processing extremely large weight data stored in files. The shape and data type of the output tensor y are inferred based on the shape and dtype attributes. The output shape is consistent with the shape attribute, and the output data type is consistent with the dtype attribute.

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
      <td>file_path</td>
      <td>Attribute</td>
      <td>String that records the file path.</td>
      <td>String, with the default value ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>file_id</td>
      <td>Attribute</td>
      <td>String that records the file ID.</td>
      <td>String, with the default value ""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Attribute</td>
      <td>Required. Shape of the data.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Required. Data type.</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>File constant tensor, whose shape is consistent with the shape attribute and whose data type is consistent with the dtype attribute.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The FileConstant operator is a basic operator of the graph engine. It is usually used as a file constant node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_file_constant](./examples/test_geir_file_constant.cpp) | Invokes the FileConstant operator by building a graph through the [operator IR](./op_graph/file_constant_proto.h) |
