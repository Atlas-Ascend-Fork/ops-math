# FillDiagonalV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: fills the diagonal of the tensor with fillValue.
- Computation formula:
  - Taking 2D as an example, when `wrap` is False, the fill positions are `[r, r]`, where `0 <= r < m` and `m = min(col, row)`. `col` is the length of the column, and `row` is the length of the row.
  - When `wrap` is True, filling is performed based on the flattened one-dimensional index: for a non-negative integer `i` that satisfies `(col + 1) * i < row * col`, the fill position is `[floor((col + 1) * i / col), ((col + 1) * i) % col]`.

## Parameter Description

<table style="table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>selfRef</td>
    <td>Input/Output tensor</td>
    <td>Input/Output tensor. Non-contiguous Tensors are supported.</td>
    <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fillValue</td>
    <td>Input</td>
    <td>Fill value. Its data type must be a data type that can be converted to FLOAT and does not overflow when converted to the data type of selfRef.</td>
    <td>Data types that can be converted to FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wrap</td>
    <td>Attribute</td>
    <td>Optional attribute. The default value is False. It indicates the fill mode and is valid only for 2D input Tensors. This parameter is ignored when the input has more than 2 dimensions. For a tall matrix (the number of rows is greater than the number of columns), if wrap is True, a new diagonal is formed every m+1 rows, where m = min(col, row).</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody>
</table>

- Atlas inference products and Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

- The number of dimensions of selfRef must be greater than 1.
- When selfRef has more than 2 dimensions, the lengths of all dimensions must be the same.
- fillValue must be convertible to the FLOAT type, and overflow must not occur when it is converted to the data type of selfRef.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_inplace_fill_diagonal](./examples/test_aclnn_fill_diagonal_v2.cpp) | Invokes the InplaceFillDiagonal operator through the [aclnnInplaceFillDiagonal](docs/aclnnInplaceFillDiagonal_en.md) API. |
