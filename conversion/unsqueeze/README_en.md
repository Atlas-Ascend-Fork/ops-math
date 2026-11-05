# Unsqueeze

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

API function: inserts a dimension with value 1 into the tensor shape, changing only the shape and not the data. Example:

When axes is empty, the output has the same shape as the input:

```text
>>> x = tf.constant([[1, 2], [3, 4]])   # shape (2, 2)
>>> unsqueeze(x).shape
(2, 2)
```

axes specifies the insertion positions:

```text
>>> x = tf.constant([[1, 2], [3, 4]])   # shape (2, 2)
>>> unsqueeze(x, [1]).shape
(2, 1, 2)
>>> unsqueeze(x, [0]).shape
(1, 2, 2)
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
      <td>Input tensor to be expanded</td>
      <td>All data types</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Optional attribute</td>
      <td><ul><li>List of positions where dimensions with value 1 are inserted. The value range is [-rank(x)-len(axes), rank(x)+len(axes))</li><li>If not specified or empty, the output shape is the same as the input shape</li><li>Duplicate values are not allowed</li><li>The default value is []</li></ul></td>
      <td>LIST_INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Expanded output tensor, whose data type is the same as that of the input x</td>
      <td>All data types</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- If axes contains duplicate values, the operator fails to execute.
- The number of output dimensions cannot exceed the maximum dimension limit (kMaxDimNum=25).
- When the input is a dynamic shape (containing -1), the -1 dimension is passed through to the output by position.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_unsqueeze](./examples/test_geir_unsqueeze.cpp) | Invokes the Unsqueeze operator by building a graph through the [operator IR](./op_graph/unsqueeze_proto.h). |
