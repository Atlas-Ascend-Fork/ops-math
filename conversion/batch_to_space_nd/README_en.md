# BatchToSpaceND

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

- Operator function: rearranges data from the batch dimension into spatial dimensions and crops the spatial dimensions.

- Function description:
  This operator is the inverse operation of SpaceToBatchND. It first rearranges the batch dimension of the input tensor into spatial dimensions according to the specified block shape (block_shape), and then crops the spatial dimensions according to the cropping parameter (crops). Specifically, it scatters data in the batch dimension into the spatial dimensions, increasing the size of the spatial dimensions while reducing the size of the batch dimension.

- Computation formula:
  Assume that the input tensor x is an N-dimensional tensor with shape $[x_0, x_1, \ldots, x_{N-1}]$, block_shape is an M-dimensional 1D tensor $[b_0, b_1, \ldots, b_{M-1}]$, and crops is an $M \times 2$ 2D tensor $[[c_{00}, c_{01}], [c_{10}, c_{11}], \ldots, [c_{M-1,0}, c_{M-1,1}]]$, satisfying $1 \leq M < N \leq 8$.

  The output tensor y has the shape $[y_0, y_1, \ldots, y_{N-1}]$, computed as follows:

  $$
  y_i = \begin{cases}
  \frac{x_0}{\prod_{j=0}^{M-1} b_j}, & i = 0 \\
  x_i \times b_{i-1} - c_{i-1,0} - c_{i-1,1}, & 1 \leq i \leq M \\
  x_i, & M+1 \leq i \leq N-1
  \end{cases}
  $$

  Here, $x_0$ must be integer-divisible by $\prod_{j=0}^{M-1} b_j$.

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
      <td>Input tensor, supporting multiple data types</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_shape</td>
      <td>Input</td>
      <td>Shape of spatial blocks, a 1D tensor with shape [M], specifying the block size of each spatial dimension</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>crops</td>
      <td>Input</td>
      <td>Cropping amounts, a 2D tensor with shape [M, 2], specifying the number of elements cropped from the top and bottom (or left and right) of each spatial dimension</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, with the same data type as the input x. The output shape is computed based on block_shape and crops</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- For <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>: BOOL is not supported.

## Constraint Description

- The dimension N of the input tensor x must satisfy 2 ≤ N ≤ 8.
- The dimension M of block_shape must satisfy 1 ≤ M < N.
- The length of block_shape must be equal to the length of the first dimension of crops.
- crops must have the shape [M, 2].
- Dimension 0 (batch dimension) of the input tensor must be integer-divisible by the product of all elements in block_shape.
- Each element in block_shape must be greater than 0.
- Each element in crops must be a non-negative integer.
- For each spatial dimension i (i = 1, 2, ..., M), the size of the dimension after cropping must be greater than or equal to 0, that is, x.shape[i] × block_shape[i-1] - crops[i-1][0] - crops[i-1][1] ≥ 0.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Graph-mode invocation | [test_geir_batch_to_space_nd](./examples/test_geir_batch_to_space_nd.cpp) | Invokes the BatchToSpaceND operator by building a graph through the [operator IR](./op_graph/batch_to_space_nd_proto.h). |
