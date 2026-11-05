# SqueezeV3

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

API function: removes dimensions whose size is 1 from a tensor. axes is provided as an optional input. Example:

If axes is empty, all dimensions whose size is 1 are removed:

```text
>>> x = torch.randn(2, 1, 3)
>>> torch.squeeze(x).shape
(2, 3)
```

If axes is specified, only the specified dimensions whose size is 1 are removed:

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
      <td>axes</td>
      <td>Optional input</td>
      <td><ul><li>List of dimensions to be removed. The value ranges from [-rank(x), rank(x))</li><li>If it is not specified or is empty, all dimensions whose size is 1 are removed</li><li>If it is specified, only the specified dimensions are removed, and the size of each specified dimension must be 1</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
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

- If axes is specified, the size of the dimension corresponding to axes must be 1 or -1. Otherwise, the operator fails to execute.
- If the value of axes is out of the range [-rank(x), rank(x)), the operator fails to execute.
- If the input has a dynamic shape (containing -1) and axes is empty, whether the -1 dimension is 1 cannot be determined, and the output shape is set to -2.
- If the input shape is -2, the output shape is -2.
- If the input shape is determined but axes cannot be obtained at the compilation stage (for example, axes comes from an upstream computation result), the output shape is set to -2.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_squeezev3](./examples/test_geir_squeezev3.cpp) | Invokes the SqueezeV3 operator by building a graph through the [operator IR](./op_graph/squeezev3_proto.h). |
