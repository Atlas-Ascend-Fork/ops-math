# Squeeze

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

API function: removes dimensions whose size is 1 from a tensor. Example:

If axis is empty, all dimensions whose size is 1 are removed:

```text
>>> x = tf.constant([[[1], [2]], [[3], [4]]])   # shape (2, 2, 1)
>>> tf.squeeze(x).shape
(2, 2)
```

If axis is specified, only the specified dimensions whose size is 1 are removed:

```text
>>> x = tf.constant([[[1], [2]], [[3], [4]]])   # shape (2, 2, 1)
>>> tf.squeeze(x, [2]).shape
(2, 2)
>>> tf.squeeze(x, [0]).shape
(2, 1)
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
      <td>BOOL, FLOAT16, FLOAT, INT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, COMPLEX64</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>List of dimensions to be removed. The value ranges from [-rank(x), rank(x))</li><li>If it is not specified or is empty, all dimensions whose size is 1 are removed</li><li>If it is specified, only the specified dimensions are removed, and the size of each specified dimension must be 1</li><li>The default value is []</li></ul></td>
      <td>LIST_INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Squeezed output tensor, whose data type is the same as that of the input x</td>
      <td>BOOL, FLOAT16, FLOAT, INT4, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, COMPLEX64</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
  </tbody></table>

## Constraint Description

- If axis is specified, the size of the dimension corresponding to axis must be 1. Otherwise, the operator fails to execute.
- If the input has a dynamic shape (containing -1) and axis is empty, whether the -1 dimension is 1 cannot be determined, and the output shape is set to -2.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_squeeze](./examples/test_geir_squeeze.cpp) | Invokes the Squeeze operator by building a graph through the [operator IR](./op_graph/squeeze_proto.h). |
