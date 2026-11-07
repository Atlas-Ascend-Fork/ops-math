# SpaceToBatchND

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: rearranges blocks of data in the spatial dimensions into the batch dimension, and pads the spatial dimensions with zeros. Multiple spatial dimensions are supported.

- Function Description:
  This operator is the inverse operation of BatchToSpaceND. It first performs zero-padding on the spatial dimensions of the input according to the paddings parameter. Then it divides the padded spatial data into blocks specified by block_shape, and moves each block to the batch dimension. The batch dimension of the output becomes the product of the elements of block_shape times that of the input, and the spatial dimensions are reduced accordingly.

- Computation formula:
  Assume that the input tensor x is an N-dimensional tensor with the shape $[x_0, x_1, \ldots, x_{N-1}]$, block_shape is an M-dimensional 1D tensor $[b_0, b_1, \ldots, b_{M-1}]$, and paddings is an $M \times 2$ 2D tensor $[[p_{00}, p_{01}], [p_{10}, p_{11}], \ldots, [p_{M-1,0}, p_{M-1,1}]]$, satisfying $1 \leq M < N \leq 8$.

  The shape of the output tensor y is $[y_0, y_1, \ldots, y_{N-1}]$, computed as follows:

  $$
  y_i = \begin{cases}
  x_0 \times \prod_{j=0}^{M-1} b_j, & i = 0 \\
  \frac{x_i + p_{i-1,0} + p_{i-1,1}}{b_{i-1}}, & 1 \leq i \leq M \\
  x_i, & M+1 \leq i \leq N-1
  \end{cases}
  $$

  For each spatial dimension i, $x_i + p_{i-1,0} + p_{i-1,1}$ must be divisible by $b_{i-1}$.

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
      <td>Input tensor, which supports multiple data types</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_shape</td>
      <td>Input</td>
      <td>Shape of the spatial block. This parameter is a 1D tensor with the shape [M], which specifies the block size of each spatial dimension</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Zero-padding amounts of the spatial dimensions. This parameter is a 2D tensor with the shape [M, 2], which specifies the number of elements padded at the head and tail of each spatial dimension</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which has the same data type as the input x. The output shape is computed based on block_shape and paddings</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>: BOOL is not supported.

## Constraint Description

1. The dimension N of the input tensor x must satisfy 2 ≤ N ≤ 8.
2. The dimension M of block_shape must satisfy 1 ≤ M < N.
3. The length of block_shape must equal the length of the first dimension of paddings.
4. The shape of paddings must be [M, 2].
5. Each element in block_shape must be greater than 0.
6. Each element in paddings must be a non-negative integer.
7. For each spatial dimension i (i = 1, 2, ..., M), the size of the padded dimension must be divisible by block_shape[i-1], that is, (x.shape[i] + paddings[i-1][0] + paddings[i-1][1]) % block_shape[i-1] == 0.
8. The operator supports empty tensors (any dimension is 0). If the number of output elements is 0, the operator returns directly without performing the move. If a spatial dimension of the input is 0 but the output is not empty after padding, the output is all zeros.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Graph-mode invocation | [test_geir_space_to_batch_nd](./examples/test_geir_space_to_batch_nd.cpp) | Invokes the SpaceToBatchND operator by building a graph through the [operator IR](./op_graph/space_to_batch_nd_proto.h). |
