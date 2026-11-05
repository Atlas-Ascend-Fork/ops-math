# MaskedFill

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: fills the elements at the positions where mask is `True` in the input Tensor `x` with specified values. `mask` must have the same shape as `x` or be broadcastable.

- Computation formula:

  $$
  y_i =
  \begin{cases}
  x_i, \, \, mask_i=False\\
  value_i, \, \, mask_i=True\\
  \end{cases}
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>Input x in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Input mask in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input value in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas training products, Atlas inference products, and Atlas 200I/500 A2 inference products: BFLOAT16 is not supported.
* Only Ascend 950PR&950DT products support INT16 and INT64.

## Constraint Description

* The dtype of value must be the same as that of x.
* x and mask must satisfy the broadcasting relationship.
* The shape of y can be obtained by broadcasting x, mask, and value together.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_inplace_masked_fill_scalar](examples/test_aclnn_inplace_masked_fill_scalar.cpp) | Invokes the MaskedFill operator through the [aclnnInplaceMaskedFillScalar](docs/aclnnInplaceMaskedFillScalar_en.md) API. |
