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
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling factor tensor. Its data type follows the constraints below, and its shape must meet the broadcast requirements.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Optional bias tensor, omitted by default. Its data type follows the constraints below. Its shape constraints are defined by the corresponding interface.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor. Its shape and data type are the same as those of x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td>Specifies the start axis for scaling. The default is 1, and the value range is [-x_rank, x_rank).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Specifies the number of axes for scaling. The default is 1, and the value range is greater than or equal to -1. If axis is negative, axis is first normalized to x_rank + axis. The following axis refers to the normalized value.</li>
          <li>When scale_from_blob is True, -1 scales through the last axis and 0 denotes scalar scaling. See the corresponding interface for shape rules.</li>
          <li>When scale_from_blob is False, num_axes does not participate in the shape derivation of scale.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_from_blob</td>
      <td>Optional attribute</td>
      <td>The default is True. True derives the shape of scale using num_axes and axis. False derives the shape from axis based on the rank of scale; num_axes does not participate in shape derivation.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

The attribute defaults apply to GE graphs. The aclnn API requires explicit axis, numAxes, and scaleFromBlob arguments, corresponding to axis, num_axes, and scale_from_blob.

<term>Atlas inference products</term>: The aclnn API does not support BFLOAT16.

## Constraint Description

- The shapes of x and y must be exactly the same.
- Each input uses FLOAT16/FLOAT/BFLOAT16, and y has the dtype of x. The aclnn API requires matching input dtypes. GE graphs allow each input to independently use these dtypes; GE adapts them before kernel dispatch.
- When bias is present and has nonzero elements, its shape must match scale. See the corresponding interface for zero-element tensor handling.
- The shape of scale must satisfy the broadcast window rules defined by axis and num_axes. See the corresponding interface for details.
- The aclnn API supports at most eight dimensions for x and scale. For GE graph rank constraints, see the [Scale IR definition](./op_graph/scale_proto.h).
- The value range of axis is [-x_rank, x_rank).
- The value range of num_axes is >=-1.
- Only the ND format is supported.
- GE graphs support -1 for unknown dimensions and [-2] for unknown rank. Runtime concrete shapes must satisfy the [Scale IR definition](./op_graph/scale_proto.h).
- The aclnn API supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) with at most eight dimensions.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_scale](./examples/test_aclnn_scale.cpp) | Invokes the Scale operator through the [aclnnScale](./docs/aclnnScale_en.md) API. |
| GE graph | - | Uses the [Scale IR definition](./op_graph/scale_proto.h); graph descriptor and runtime shape constraints are defined there. |

<term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term> support GE graphs but do not support the aclnn API.
