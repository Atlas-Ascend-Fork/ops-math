# PadV3GradReplication

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: backpropagation of replication_pad3d.

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
    <td>Forward propagation of replication_pad3d. The shape supports 4 or 5 dimensions, and the dimensions must be consistent with self and gradInput.</td>
    <td>FLOAT16, FLOAT32, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input</td>
    <td>The shape supports 4 or 5 dimensions, and the dimensions must be consistent with gradOutput and gradInput. The shape is the same as that of gradInput.</td>
    <td>FLOAT16, FLOAT32, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input</td>
    <td>The length of padding is 6, indicating the values to be padded on the left, right, top, bottom, front, and back of the 3D tensor. The first two values must be less than the size of the last dimension of self; the third and fourth values must be less than the size of the second-to-last dimension of self; the fifth and sixth values must be less than the size of the third-to-last dimension of self.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output</td>
    <td>The shape supports 4 or 5 dimensions, and the dimensions and data type are the same as those of self.</td>
    <td>FLOAT16, FLOAT32, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- The data types of gradOutput, self, and gradInput must be the same.
- The dimensions of gradOutput, self, and gradInput support only 4 or 5.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_pad_v3_grad_replication](examples/test_aclnn_pad_v3_grad_replication.cpp) | Invokes the PadV3GradReplication operator through the [aclnnReplicationPad3dBackward](docs/aclnnReplicationPad3dBackward_en.md) API.    |
