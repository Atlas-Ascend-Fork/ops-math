# SpaceToBatch

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                             | √        |
| <term>Atlas A3 products</term>     | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             | √        |
| <term>Atlas training products</term>                              | √        |

## Function Description

- Operator function: rearranges blocks of data in the spatial dimensions into the batch dimension, and pads the spatial dimensions with zeros.

- Function Description:
  This operator first performs zero-padding on the spatial dimensions of the input according to the paddings parameter. Then it divides the padded spatial data into blocks of `block_size × block_size`, and moves each block to the batch dimension. The batch dimension of the output becomes `block_size × block_size` times that of the input, and the spatial dimensions are reduced accordingly.

- Computation formula:
  Assume that the input x is a 4D NHWC tensor `[N, H_in, W_in, C]`, block_size = bs, and paddings = `[[pad_top, pad_bottom], [pad_left, pad_right]]`.

  H_padded = H_in + pad_top + pad_bottom, W_padded = W_in + pad_left + pad_right,

  H_out = H_padded / bs, W_out = W_padded / bs.

  H_padded and W_padded must both be divisible by bs.

  The shape of the output y is `[N * bs * bs, H_out, W_out, C]`.

  Coordinate mapping (output → input):

  ```text
  n     = n_out / (bs * bs)
  bh    = (n_out % (bs * bs)) / bs
  bw    = n_out % bs
  h_in  = h_out * bs + bh - pad_top
  w_in  = w_out * bs + bw - pad_left

  If 0 ≤ h_in < H_in and 0 ≤ w_in < W_in:
    y[n_out, h_out, w_out, c] = x[n, h_in, w_in, c]
  Otherwise:
    y[n_out, h_out, w_out, c] = 0
  ```

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
      <td>Input tensor, which is a 4D NHWC tensor [N, H_in, W_in, C] and supports multiple data types</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Zero-padding amounts of the spatial dimensions. This parameter is a 2D tensor with the shape [2, 2] and the value [[pad_top, pad_bottom], [pad_left, pad_right]]</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute (Required)</td>
      <td>Size of the spatial block. This parameter must be an integer greater than 0. H_padded and W_padded must be divisible by block_size</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which has the same data type as the input x. The output shape is [N*block_size*block_size, H_out, W_out, C]</td>
      <td>Same as x</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- The input tensor x must be in 4D NHWC format.
- block_size must be greater than 0.
- The shape of paddings is [2, 2], and each element must be >= 0.
- H_padded = H_in + pad_top + pad_bottom must be divisible by block_size.
- W_padded = W_in + pad_left + pad_right must be divisible by block_size.
- block_size is a compile-time constant (operator attribute), not a runtime tensor input.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  Graph-mode invocation  |[test_geir_space_to_batch](examples/test_geir_space_to_batch.cpp) | Invokes the SpaceToBatch operator by building a graph through the [operator IR](./op_graph/space_to_batch_proto.h). |
