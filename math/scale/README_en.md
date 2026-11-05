# Scale

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |   √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: performs the scale and bias computation on the input tensor. If bias is not input, $y = x \cdot scale$. If bias is input, $y = x \cdot scale + bias$.
- Operator formula:

  If bias is not input:

  $$
  y_i = x_i \cdot scale_i
  $$

  If bias is input:

  $$
  y_i = x_i \cdot scale_i + bias_i
  $$

  Here:
  - $x_i$ is the input tensor;
  - $scale_i$ is the scaling factor tensor, and broadcast with x is supported;
  - $bias_i$ is an optional bias tensor, and its shape is the same as that of scale;
  - $y_i$ is the output tensor, and its shape is the same as that of x.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>Input tensor of the operator. Empty tensors are supported.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling factor tensor. Its data type must be the same as that of x, and its shape must meet the broadcast requirements.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Bias tensor. If it is not empty, its data type must be the same as that of scale, and its shape must be the same as that of scale.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Specifies the start axis for scaling. The value range is [-x_rank, x_rank).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>Attribute</td>
      <td>
        <ul>
          <li>Specifies the number of axes for scaling. The value range is greater than or equal to -1. If axis is negative, axis is first normalized to x_rank + axis. The following axis refers to the normalized value.</li>
          <li>When scale_from_blob is True: num_axes=-1 indicates scaling from the axis axis to the last axis, and the rank of scale must be x_rank - axis. num_axes=0 indicates that scale is a 0-dimensional tensor. When num_axes is greater than 0, axis + num_axes cannot be greater than x_rank, and the rank of scale must be num_axes. Each dimension of scale must be equal to the corresponding dimension of x starting from axis.</li>
          <li>When scale_from_blob is False, num_axes does not participate in the shape derivation of scale.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_from_blob</td>
      <td>Attribute</td>
      <td>True: derives the shape of scale using numAxes + axis. False: derives the shape from axis based on the rank of scale, and ignores numAxes.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor. Its shape and data type are the same as those of x.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The shapes of x and y must be exactly the same.
- The dtypes of x, scale, bias, and y must be the same (all are one of FLOAT16/FLOAT32/BFLOAT16).
- If bias is not empty, the shape of bias must be the same as that of scale.
- The shape of scale must meet the broadcast rules (see Function Description).
- The number of shape dimensions of x and scale is not greater than 8.
- The value range of axis is [-x_rank, x_rank).
- The value range of numAxes is >=-1.
- Only the ND format is supported.
- The GE IR dynamic rank scenario (the input shape is declared as `[-2]`) is not supported currently.
- [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the number of dimensions of a non-contiguous tensor is not greater than 8.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_scale](./examples/test_aclnn_scale.cpp) | Invokes the Scale operator through the [aclnnScale](./docs/aclnnScale_en.md) API. |
