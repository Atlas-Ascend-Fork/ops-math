# ConstPlaceHolder

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

Creates a constant placeholder tensor. This operator processes inputs whose addresses remain unchanged but whose values are uncertain. Converting an input to this type helps improve the graph execution performance (prevents input addresses from being repeatedly refreshed). The shape and data type of the output tensor y are inferred based on the origin_shape and dtype attributes. The output shape is consistent with the origin_shape attribute, and the output data type is consistent with the dtype attribute.

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
      <td>origin_shape</td>
      <td>Attribute</td>
      <td>Required. Original shape of the tensor.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>origin_format</td>
      <td>Attribute</td>
      <td>Required. Original format of the tensor.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storage_shape</td>
      <td>Attribute</td>
      <td>Required. Storage shape of the tensor.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storage_format</td>
      <td>Attribute</td>
      <td>Required. Storage format of the tensor.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expand_dim_rules</td>
      <td>Attribute</td>
      <td>Required. Dimension expansion rules for converting origin_shape, origin_format, and storage_format to storage_shape.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Required. Data type of the tensor.</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>addr</td>
      <td>Attribute</td>
      <td>Required. Address of the tensor.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Attribute</td>
      <td>Required. Size of the address.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>placement</td>
      <td>Attribute</td>
      <td>Placement position of the tensor. 0 indicates Host, and 1 indicates Device.</td>
      <td>Int, with the default value 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Constant placeholder tensor, whose shape is consistent with the origin_shape attribute and whose data type is consistent with the dtype attribute.</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

None

## Invocation Description

The ConstPlaceHolder operator is a basic operator of the graph engine. It is usually used as a constant placeholder node in a graph and is not invoked separately.

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_const_place_holder](./examples/test_geir_const_place_holder.cpp) | Invokes the ConstPlaceHolder operator by building a graph through the [operator IR](./op_graph/const_place_holder_proto.h) |
