# SqueezeV2

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

Operator function: removes dimensions whose size is 1 from a tensor. axis is provided as an attribute. This operator is a deprecated version, and it is recommended that you use the Squeeze operator first. Example:

If axis is empty, all dimensions whose size is 1 are removed:

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x).shape
(2, 3)
```

If axis is specified, only the specified dimensions whose size is 1 are removed:

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x, 1).shape
(2, 3)
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
      <td>Input tensor to be squeezed</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td><ul><li>List of dimensions to be removed. The value ranges from [-rank(x), rank(x)), and negative indices are supported</li><li>If it is not specified or is empty, all dimensions whose size is 1 are removed</li><li>If it is specified, only the specified dimensions are removed, and the size of each specified dimension must be 1</li></ul></td>
      <td>ListInt, and the default value is []</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Squeezed output tensor, whose data type is the same as that of the input x</td>
      <td>All types</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- If axis is specified, the size of the dimension corresponding to axis must be 1 or -1. Otherwise, the operator fails to execute.
- If the value of axis is out of the range [-rank(x), rank(x)), or the number of axes after deduplication based on the original values is greater than or equal to the number of input dimensions, the output shape falls back to the input shape (squeeze is not executed).
- If the input has a dynamic shape (containing -1) and axis is empty, whether the -1 dimension is 1 cannot be determined, and the output shape is set to -2.
- If the input shape is -2, the output shape is -2.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_squeezev2](./examples/test_geir_squeezev2.cpp) | Invokes the SqueezeV2 operator by building a graph through the [operator IR](./op_graph/squeezev2_proto.h). |
