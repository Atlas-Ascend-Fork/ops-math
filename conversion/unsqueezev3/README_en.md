# UnsqueezeV3

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT products</term>       |   √  |
| <term>Atlas A3 products</term> |   √  |
| <term>Atlas A2 products</term> |   √  |
| <term>Atlas 200I/500 A2 inference products</term>          |   √  |
| <term>Atlas inference products</term>                    |   √  |
| <term>Atlas training products</term>                    |   √  |

## Function Description

API function: inserts dimensions with value 1 at specified positions of the input tensor, changing only the shape and not the data. axes is provided as an input. Example:

When axes is specified, dimensions of 1 are inserted at the corresponding positions:

```text
>>> x = torch.randn(2, 3)
>>> torch.unsqueeze(x, 1).shape
(2, 1, 3)
```

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
      <td>Input tensor for dimension insertion</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Input</td>
      <td><ul><li>List of positions where dimensions are to be inserted. The value range is [-(rank(x)+len(axes)), rank(x)+len(axes))</li><li>Values in the list cannot be duplicated</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after dimension insertion, whose data type is the same as that of the input x</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- If the value of axes exceeds the range [-(rank(x)+len(axes)), rank(x)+len(axes)), the operator fails to execute.
- If axes contains duplicate values, the operator fails to execute.
- When axes is empty, the output shape is the same as that of the input x.
- When the axes data cannot be obtained in the compilation phase (for example, axes comes from an upstream computation result), the output shape is set to -2.
- When the input shape is -2, the output shape is -2.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_unsqueezev3](./examples/test_geir_unsqueezev3.cpp) | Invokes the UnsqueezeV3 operator by building a graph through the [operator IR](./op_graph/unsqueezev3_proto.h). |
