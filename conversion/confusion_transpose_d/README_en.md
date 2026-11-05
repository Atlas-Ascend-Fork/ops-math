# ConfusionTransposeD

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|

## Function Description

- Operator function: fuses the reshape and transpose operations.

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>perm</td>
      <td>Input</td>
      <td>Index of the axis before transposition corresponding to each axis after transposition.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Input</td>
      <td>Shape after reshape.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transposeFirst</td>
      <td>Input</td>
      <td>Determines whether the transpose operation is performed first.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Same as the input x.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

-Elements in the perm tensor must be unique and within the range [0, number of dimensions of perm - 1].
-When transposeFirst is True, the length of perm must be the same as the length of the shape of x, that is, len(perm)=len(x_shape); when transposeFirst is False, the length of perm must be the same as the length of the attribute input shape, that is, len(perm)=len(shape).
-The product of all dimensions in shape must be equal to the total number of elements in the input tensor x.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_confusion_transpose.cpp](examples/arch35/test_aclnn_confusion_transpose.cpp) | Invokes the ConfusionTransposeD operator through the [aclnnConfusionTranspose](./docs/aclnnConfusionTranspose_en.md) API. |
