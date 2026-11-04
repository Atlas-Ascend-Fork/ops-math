# CircularPadGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √      |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: backward propagation operation of CircularPad.
- CircularPad2dBackward: computes the backward propagation of CircularPad2d.
- CircularPad3dBackward: computes the backward propagation of CircularPad3d.

## Parameter Description

### CircularPad2dBackward

<table style="table-layout: fixed; width: 1007px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 284px">
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
    <td>gradOutput</td>
    <td>Input tensor</td>
    <td>Gradient data input during backward propagation, whose shape must be the same as the output of the circular_pad2d forward propagation.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input tensor</td>
    <td>Original input data to be padded during forward propagation, whose shape is the same as gradInput.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input array</td>
    <td>Padding dimensions during forward propagation, with a length of 4. The values sequentially represent the padding values for left, right, top, and bottom.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output tensor</td>
    <td>Gradient data output during backward propagation, whose shape is the same as self.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### CircularPad3dBackward

<table style="table-layout: fixed; width: 1007px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 284px">
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
    <td>gradOutput</td>
    <td>Input tensor</td>
    <td>Gradient data input during backward propagation, whose shape must be the same as the output of the circular_pad3d forward propagation.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input tensor</td>
    <td>Original input data to be padded during forward propagation, whose shape is the same as gradInput.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input array</td>
    <td>Padding dimensions during forward propagation, with a length of 6. The values sequentially represent the padding values for left, right, top, bottom, front, and back.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output tensor</td>
    <td>Gradient data output during backward propagation, whose shape is the same as self.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- The size of the last dimension of gradOutput under different types must meet the following constraints:
  - float16/bfloat16: (0, 16384)
  - float32: (0, 24576)
- Each padding value must be less than the size of the corresponding dimension.
- The data types of the input and output must be the same.
- The shape of gradOutput must be the same as the output of the forward propagation.
- The shape of gradInput must be the same as self.

## Invocation Description

| Invocation Method  | Sample Code                                                  | Description                                                         |
| --------- |-------------------------------------------------------| ------------------------------------------------------------ |
| aclnn API | [test_aclnn_circular_pad_grad](./examples/test_aclnn_circular_pad_grad.cpp)       | Invokes the CircularPad2dBackward operator through the [aclnnCircularPad2dBackward](docs/aclnnCircularPad2dBackward_en.md) API. |
| aclnn API | [test_aclnn_circular_pad3d_backward](./examples/test_aclnn_circular_pad3d_backward.cpp) | Invokes the CircularPad3dBackward operator through the [aclnnCircularPad3dBackward](docs/aclnnCircularPad3dBackward_en.md) API. |
