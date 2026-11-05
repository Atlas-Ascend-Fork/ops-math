# BatchToSpace

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                             | √        |
| <term>Atlas A3 products</term>     | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>                      |     √     |
| <term>Atlas inference products</term>                             | √        |
| <term>Atlas training products</term>                              | √        |

## Function Description

- Operator function: rearranges data from the batch dimension into spatial dimensions and crops the spatial dimensions.

- Function description:
  This operator is the inverse operation of SpaceToBatch. The input is a 4D NHWC tensor `[N * bs * bs, H_in, W_in, C]`, and the output is a 4D NHWC tensor `[N, H_out, W_out, C]`. The operator restores the `bs × bs` spatial blocks folded in the input batch dimension to the H and W dimensions, and crops the spatial boundaries through the crops parameter.

- Computation formula:
  Assume that the input x is a 4D NHWC tensor `[N_in, H_in, W_in, C]`, block_size = bs, and crops = `[[crop_top, crop_bottom], [crop_left, crop_right]]`.

  N_out = N_in / (bs × bs), H_out = H_in × bs - crop_top - crop_bottom, and W_out = W_in × bs - crop_left - crop_right.

  N_in must be integer-divisible by bs × bs, and the cropped H_out and W_out must be greater than 0.

  The output y has the shape `[N_out, H_out, W_out, C]`.

## Parameter Description

<table style="table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, a 4D NHWC tensor [N*bs*bs, H_in, W_in, C], supporting multiple data types</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, UINT8, UINT16, INT64, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>crops</td>
      <td>Input</td>
      <td>Cropping amounts of spatial dimensions, a 2D tensor with shape [2, 2] and values [[crop_top, crop_bottom], [crop_left, crop_right]]</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute</td>
      <td>Size of the spatial block, which must be an integer greater than 0. N_in must be integer-divisible by block_size × block_size</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, with the same data type as the input x. The output shape is [N_out, H_out, W_out, C]</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- The input tensor x must be a 4D tensor (logical layout NHWC, storage format ND), with shape [N * bs * bs, H_in, W_in, C].
- block_size must be greater than 0.
- The input batch dimension N_in must be integer-divisible by block_size × block_size.
- crops has the shape [2, 2], and each element must be >= 0.
- The spatial dimensions after cropping cannot be negative: H_in × block_size - crop_top - crop_bottom >= 0 and W_in × block_size - crop_left - crop_right >= 0 (a value of 0 indicates that the output of the spatial dimension is an empty tensor).

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  Graph mode  |[test_geir_batch_to_space](examples/test_geir_batch_to_space.cpp) | Invokes the BatchToSpace operator by building a graph through the [operator IR](./op_graph/batch_to_space_proto.h). |
