# ReflectionPad3dGrad

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

- Operator function: computes the backpropagation of the aclnnReflectionPad3d API.

## Parameter Description

<table style="table-layout: fixed; width: 866px"><colgroup>
<col style="width: 139px">
<col style="width: 148px">
<col style="width: 236px">
<col style="width: 267px">
<col style="width: 76px">
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
    <td>Input of backpropagation.</td>
    <td>FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input</td>
    <td>Input tensor of forward propagation.</td>
    <td>FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input</td>
    <td>The length is 6. The values represent the values to be padded on the left, right, top, bottom, front, and back in sequence.</td>
    <td>aclIntArray array</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output</td>
    <td>Output of backpropagation.</td>
    <td>FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- The dimensions of gradOutput, self, and gradInput must be consistent (four or five dimensions are supported), and their shapes must be consistent with the output shape of the forward propagation of reflection_pad3d.
- Input value range restriction: the first two values of padding must be smaller than the value of the last dimension of self, the middle two values must be smaller than the value of the second-to-last dimension of self, and the last two values must be smaller than the value of the third-to-last dimension of self.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_reflection_pad3d_grad](./examples/test_aclnn_reflection_pad3d_grad.cpp) | Invokes the ReflectionPad3dGrad operator through the [aclnnReflectionPad3dBackward](docs/aclnnReflectionPad3dBackward_en.md) API. |
