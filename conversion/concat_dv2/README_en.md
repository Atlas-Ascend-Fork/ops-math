# ConcatDV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|Ascend 950PR&950DT products|x|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|x|
|Atlas inference products|x|
|Atlas training products|x|

## Function Description

- Operator function: concatenates multiple input tensors along the specified dimension, and outputs a tensor containing all input data concatenated in order.
- Computation process:
  - Input:
    - Tensor list x[0], x[1],..., x[N-1]
    - Concatenation dimension concat_dim

  - Process:
    1. Check that all input tensors have the same data type;
    2. Check that all dimensions except concat_dim are identical;
    3. Concatenate along the concat_dim dimension in sequence:
       y = ConcatDV2(x[0], x[1], ..., x[N-1], axis = concat_dim)
  - Output: concatenated tensor y

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 930px"><colgroup>
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 270px">
  <col style="width: 320px">
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
      <td>Dynamic input list, which is the input x[i] in the process diagram.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, UINT8, INT16, INT8, COMPLEX64, INT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>Input</td>
      <td>Specifies the concatenation dimension, which is concat_dim in the computation process.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the computation process.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, UINT8, INT16, INT8, COMPLEX64, INT64, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- All input tensors must have the same shape except for the concatenation dimension.
- The concatenation dimension concat_dim currently supports only 0.
- All tensors in x must have the same data type.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_cat](../concat_d/examples/test_aclnn_cat.cpp) | Invokes the ConcatDV2 operator through the [aclnnCat](../concat_d/docs/aclnnCat_en.md) API. |
