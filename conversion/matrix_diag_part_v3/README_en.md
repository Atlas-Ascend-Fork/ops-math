# MatrixDiagPartV3

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

- Operator function: extracts one or more diagonals from the last two dimensions of the input matrix or batch of matrices, and pads shorter diagonals with $\mathrm{padding\_value}$ in the manner specified by `align`.
- Assume that the sizes of the last two dimensions of input $x$ are $M$ and $N$, and $k=[k_l, k_u]$ indicates the range of diagonals to extract. Let:

  $$
  \mathrm{num\_diags} = k_u - k_l + 1
  $$

  $$
  \mathrm{max\_diag\_len} = \min(M + \min(k_u, 0),\ N - \max(k_l, 0))
  $$

  For the diagonal numbered $d$, its valid length is:

  $$
  \mathrm{diag\_len}(d) = \min(M + \min(d, 0),\ N - \max(d, 0))
  $$

  Output elements satisfy:

  $$
  y_{..., m, n} =
  \begin{cases}
  x_{..., n + \max(-d, 0),\ n + \max(d, 0)} & 0 \le n < \mathrm{diag\_len}(d) \\
  \text{padding\_value} & \text{otherwise}
  \end{cases}
  $$

  In the banded-output scenario, $d = k_u - m$; in the single-diagonal scenario, $d = k_l = k_u$. When $k$ is a single integer or $k_l = k_u$, the output shape is `[..., max_diag_len]`; when $k_l < k_u$, the output shape is `[..., num_diags, max_diag_len]`.

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
      <td>`x` in the formula. The last two dimensions represent the matrices from which diagonals are extracted, and the other leading dimensions are processed as batch dimensions.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>k<sub>l</sub> and k<sub>u</sub> in the formula. It can be a scalar, indicating that a single diagonal is extracted; it can also be a vector of length 2, indicating that a diagonal band is extracted.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding_value</td>
      <td>Input</td>
      <td>`padding_value` in the formula. It is used to pad invalid positions of shorter diagonals, and its data type is the same as that of `x`.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
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
      <td>`y` in the formula, indicating the extracted diagonal tensor. Its data type is the same as that of `x`.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The rank of `x` is at least 2.
- The number of elements in `k` can only be 1 or 2; when `k` has 2 elements, `k[0] <= k[1]` must be satisfied.
- When `k` indicates a diagonal band, the length of the second-to-last dimension of the output is $k_u - k_l + 1$, and the length of the last dimension is $\mathrm{max\_diag\_len}$.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_matrix_diag_part_v3](./examples/test_geir_matrix_diag_part_v3.cpp) | Invokes the MatrixDiagPartV3 operator by building a graph through the [operator IR](./op_graph/matrix_diag_part_v3_proto.h). |
