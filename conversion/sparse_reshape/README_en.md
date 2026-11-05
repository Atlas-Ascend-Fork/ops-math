# SparseReshape

English | [简体中文](./README.md)

## Contribution Description

| Contributor      | Contributed Operator | Contribution Date       | Contribution Content     |
|----------|------|------------|----------|

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function Description

- Operator function: reshapes the indices of a sparse tensor (SparseTensor) from the input shape to the target shape. A sparse tensor consists of indices (coordinates of non-zero elements), values (values of non-zero elements), and shape (dense shape). SparseReshape reshapes only the indices and shape, and does not modify the values.

- Computation formula:

$$
\text{flat\_id} = \sum_{j=0}^{\text{input\_rank}-1} \text{indices}[i, j] \times \text{input\_strides}[j]
$$

$$
\text{y\_indices}[i, j] = \lfloor \text{flat\_id} / \text{output\_strides}[j] \rfloor, \quad \text{flat\_id} = \text{flat\_id} \bmod \text{output\_strides}[j]
$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>indices</td>
      <td>Input</td>
      <td>Multi-dimensional coordinate matrix of non-zero elements. This parameter is a 2D tensor with the shape (nnz, input_rank).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Input</td>
      <td>Original dense shape. This parameter is a 1D tensor with the shape (input_rank,).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new_shape</td>
      <td>Input</td>
      <td>Target dense shape. This parameter is a 1D tensor with the shape (output_rank,). One value can be -1 and is inferred automatically.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_indices</td>
      <td>Output</td>
      <td>Reshaped multi-dimensional coordinate matrix of non-zero elements. This parameter is a 2D tensor with the shape (nnz, output_rank).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_shape</td>
      <td>Output</td>
      <td>Reshaped dense shape. This parameter is a 1D tensor with the shape (output_rank,).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input indices must be a 2D matrix, and shape and new_shape must be 1D vectors.
- Each row of the input indices contains the multi-dimensional coordinates of a non-zero element. The value of each element must be greater than or equal to 0 and less than the size of the corresponding dimension in shape.
- Both input_rank and output_rank do not exceed 8 dimensions (MAX_RANK=8).
- At most one value in new_shape can be -1. This value is inferred automatically by dividing the product of the values of the shape parameter by the product of the other non--1 values of the new_shape parameter.
- The product of the values of the input shape must be the same as the product of the values of the input new_shape.
- The dtypes of all inputs and outputs must be the same (all int32 or all int64).

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation Method</th>
    <th>Invocation Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_sparse_reshape.cpp">test_geir_sparse_reshape</a></td>
    <td>For operator compilation and verification, refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
</tbody>
</table>
