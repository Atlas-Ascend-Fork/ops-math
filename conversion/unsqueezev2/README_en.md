# UnsqueezeV2

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

Operator function: inserts a dimension with value 1 into the tensor shape. axis is provided as an attribute. This operator is a deprecated version. It is recommended that you use the Unsqueeze operator instead. Example:

When axis is specified, a dimension with value 1 is inserted at the specified position:

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, 1).shape
(2, 1, 3)
```

axis supports negative indices, where -1 indicates insertion at the end:

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, -1).shape
(2, 3, 1)
```

When axis is empty or not specified, no dimension is inserted, and the output shape is the same as the input shape.

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1528px"><colgroup>
  <col style="width: 132px">
  <col style="width: 120px">
  <col style="width: 256px">
  <col style="width: 253px">
  <col style="width: 333px">
  <col style="width: 126px">
  <col style="width: 160px">
  <col style="width: 145px">
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
      <td>Input tensor for dimension expansion</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td><ul><li>List of positions where dimensions are to be inserted. The value range is [-rank(x)-len(axis), rank(x)+len(axis)), and negative indices are supported</li><li>Positions cannot be duplicated (determined by the positions normalized from negative indices)</li><li>If not specified or empty, no dimension is inserted, and the output shape is the same as the input shape</li></ul></td>
      <td>ListInt, with the default value of []</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after dimension expansion, whose data type is the same as that of the input x</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- If the value of axis exceeds the range [-rank(x)-len(axis), rank(x)+len(axis)), or the normalized positions are duplicated, the operator fails to execute.
- If the number of output dimensions rank(x)+len(axis) exceeds the maximum dimension limit of the framework, the operator fails to execute.
- When axis is empty or not specified, the output shape is the same as the input shape.
- When the input shape is -2, the output shape is -2.
- When the input contains -1 (unknown dimension), the unknown dimension is retained as is.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_unsqueezev2](./examples/test_geir_unsqueezev2.cpp) | Invokes the UnsqueezeV2 operator by building a graph through the [operator IR](./op_graph/unsqueezev2_proto.h). |
