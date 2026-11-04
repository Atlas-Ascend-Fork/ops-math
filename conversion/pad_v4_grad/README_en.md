# PadV4Grad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    ×     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: backpropagation of reflection_pad1d/reflection_pad2d.

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 168.333333px">
<col style="width: 116.666666px">
<col style="width: 305.666666px">
<col style="width: 233.666666px">
<col style="width: 97.666666px">
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
    <td>gradOutput</td>
    <td>Input</td>
    <td>Forward propagation of reflection_pad1d/reflection_pad2d. The shape supports 2 to 4 dimensions, and the dimensions must be consistent with self and gradInput. The backward operator of reflection_pad1d corresponds to 2/3 dimensions, and the backward operator of reflection_pad2d corresponds to 3/4 dimensions.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input</td>
    <td>The shape supports 2 to 4 dimensions, and the dimensions must be consistent with self and gradInput. The shape is the same as that of gradInput.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input</td>
    <td>padding describes the size of outward padding, and its length is 2 or 4.<li>When the length is 2, it indicates the values to be padded on the left and right for 1d, and the first two values of padding must be smaller than the value of the last dimension of self.</li><li>When the length is 4, it indicates the values to be padded on the left, right, top, and bottom for 2d. The first two values of padding must be smaller than the value of the last dimension of self, and the last two values must be smaller than the value of the second-to-last dimension.</li></td>
    <td>INT64</td>
    <td></td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output</td>
    <td>The data type is the same as that of self.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- Input shape restriction: the dimensions of gradOutput, self, and gradInput must be consistent (three or four dimensions are supported), and their shapes must be consistent with the output shape of the forward propagation of reflection_pad1d/reflection_pad2d.

- Input value range restriction: when the length of padding is 2, the first two values of padding must be smaller than the value of the last dimension of self; when the length is 4, the first two values of padding must be smaller than the value of the last dimension of self, and the last two values must be smaller than the value of the second-to-last dimension.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_reflection_pad1d_backward.cpp](examples/test_aclnn_reflection_pad1d_backward.cpp) | Invokes the PadV4Grad operator through the [aclnnReflectionPad1dBackward](docs/aclnnReflectionPad1dBackward_en.md) API.    |
