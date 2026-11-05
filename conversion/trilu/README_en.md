# Trilu

English | [简体中文](./README.md)

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

- Operator function: for the matrix formed by the last two dimensions of the input tensor, retains the elements in the upper or lower triangular region based on the `upper` attribute, and sets the other elements to zero.

- Computation formula:

$$
\text{When upper = 1 (upper triangular mode):} \quad y[row, col] = \begin{cases} x[row, col] & \text{if } col - row \ge diagonal \\ 0 & \text{otherwise} \end{cases}
$$

$$
\text{When upper = 0 (lower triangular mode):} \quad y[row, col] = \begin{cases} x[row, col] & \text{if } col - row \le diagonal \\ 0 & \text{otherwise} \end{cases}
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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the triangular mask computation. The rank is at least 2, and the last two dimensions form a matrix.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input (optional)</td>
      <td>Diagonal offset. 0 indicates the main diagonal, a positive value indicates the superdiagonal, and a negative value indicates the subdiagonal. The default value is 0.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>upper</td>
      <td>Attribute</td>
      <td>Triangular region flag. 1 indicates the upper triangle, and 0 indicates the lower triangle. The default value is 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after the triangular mask computation, whose shape is exactly the same as that of the input x.</td>
      <td>Same as the input x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Ascend 950PR&950DT products</term>: The supported data types are UINT64, INT64, UINT32, INT32, UINT16, INT16, UINT8, INT8, FLOAT16, FLOAT32, DOUBLE, BOOL, BFLOAT16, COMPLEX32, and COMPLEX64.

## Constraint Description

- The input `x` must satisfy `rank >= 2` (ND format), and the last two dimensions form a matrix.
- 0D and 1D inputs are not supported. Such inputs return `GRAPH_FAILED` in the Tiling phase.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_trilu](./examples/test_geir_trilu.cpp) | Invokes the trilu operator by building a graph through the GE IR on the host side. |
