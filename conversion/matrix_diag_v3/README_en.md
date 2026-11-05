# MatrixDiagV3

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: generates matrices from the input values of one or more diagonals, and positions outside the diagonal band are filled with $\mathrm{padding\_value}$.
- Assume that the sizes of the last two dimensions of the output tensor are $\mathrm{num\_rows}$ and $\mathrm{num\_cols}$ respectively, and $k=[k_l, k_u]$ indicates the range of diagonals to write. In the single-diagonal scenario, $k_l = k_u$. Let $d = j - i$ indicate the diagonal number of position $(i, j)$. Then the maximum diagonal length is:

$$
\mathrm{max\_diag\_len} = \min(\mathrm{num\_rows} + \min(k_u, 0),\ \mathrm{num\_cols} - \max(k_l, 0))
$$

  Output elements satisfy:

  $$
  y_{..., i, j} =
  \begin{cases}
  x_{..., k_u - d,\ p(i, j)} & k_l \le d \le k_u \\
  \text{padding\_value} & \text{otherwise}
  \end{cases}
  $$

  Here, $p(i, j)$ indicates the position of the diagonal element in the last dimension of input $x$, and the left/right alignment mode is controlled by the `align` attribute. When $k$ is a single integer or $k_l = k_u$, $x$ indicates a single diagonal; when $k_l < k_u$, $x$ indicates a diagonal band.

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
      <td>x</td>
      <td>Input</td>
      <td>`x` in the formula. When `k` indicates a single diagonal, the last dimension of `x` stores the data of this diagonal; when `k` indicates a diagonal band, the second-to-last dimension of `x` stores the number of diagonals, and the last dimension stores the data of each diagonal padded according to `align`.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>k<sub>l</sub> and k<sub>u</sub> in the formula. It can be a scalar, indicating a single diagonal; it can also be a vector of length 2, indicating the lower bound and upper bound of a diagonal band.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_rows</td>
      <td>Input</td>
      <td>Number of rows of the output matrix, which is `num_rows` in the formula. A value of `-1` indicates that it is automatically inferred from `k` and `x`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_cols</td>
      <td>Input</td>
      <td>Number of columns of the output matrix, which is `num_cols` in the formula. A value of `-1` indicates that it is automatically inferred from `k` and `x`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding_value</td>
      <td>Input</td>
      <td>`padding_value` in the formula. It is used to fill positions outside the specified diagonal band, and its data type is the same as that of `x`.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>align</td>
      <td>Optional attribute</td>
      <td>Specifies the alignment mode of superdiagonals and subdiagonals. `RIGHT_LEFT`, `LEFT_RIGHT`, `LEFT_LEFT`, and `RIGHT_RIGHT` are supported, and the default value is `RIGHT_LEFT`.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>`y` in the formula, indicating the generated matrix tensor. Its data type is the same as that of `x`. When `k` is a single element or `k[0] == k[1]`, the rank of `y` is the rank of `x` plus 1; otherwise, the rank of `y` is the same as that of `x`.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The rank of `x` is at least 1.
- The number of elements in `k` can only be 1 or 2; when `k` has 2 elements, `k[0] <= k[1]` must be satisfied.
- When `k` indicates a diagonal band, the length of the second-to-last dimension of `x` must be equal to $k_u - k_l + 1$, and the length of the last dimension must be equal to $\mathrm{max\_diag\_len}$.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_matrix_diag_v3](./examples/test_geir_matrix_diag_v3.cpp) | Invokes the MatrixDiagV3 operator by building a graph through the [operator IR](./op_graph/matrix_diag_v3_proto.h). |
