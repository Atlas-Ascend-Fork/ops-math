# Tril

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |     √      |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |   ×     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

  - Operator function: sets the part above the diagonal in the last two dimensions (counted from left to right in the shape) of the input self tensor to zero. The parameter diagonal can be positive or negative, and the default value is 0. A positive value moves the main diagonal toward the upper right, and a negative value moves the main diagonal toward the lower left.
  - Computation formula: In the following, i denotes the index for traversing the elements of the second-to-last dimension (i is the row index), j denotes the index for traversing the elements of the last dimension (j is the column index), and d denotes diagonal. In the 2D coordinate map corresponding to (i, j), i+d==j indicates that the position is on the diagonal.

    $$
    On and below the diagonal, that is, i+d>=j, the original value is retained: out_{i, j} = self_{i, j}\\
    Above the diagonal, that is, i+d<j, the value is set to zero (excluding the diagonal): out_{i, j} = 0
    $$

  - Example:

    $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$,
    the result of triu(self, diagonal=0) is:
    $\begin{bmatrix} [9&0&0] \\ [1&2&0] \\ [3&4&1] \end{bmatrix}$;
    with a different value of diagonal, the result of triu(self, diagonal=1) is:
    $\begin{bmatrix} [9&6&0] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$;
    with diagonal set to -1, the result of triu(self, diagonal=-1) is:
    $\begin{bmatrix} [0&0&0] \\ [1&0&0] \\ [3&4&0] \end{bmatrix}$.

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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the tril computation, which is self in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>Input</td>
      <td>Input parameter for the tril computation, which is diagonal in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the tril computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>: The supported data types are DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, and BOOL.
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The supported data types are DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL, and BFLOAT16.
  - <term>Ascend 950PR&950DT products</term>: The supported data types are DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL, BFLOAT16, COMPLEX32, and COMPLEX64.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_tril](./examples/test_aclnn_tril.cpp) | Invokes the Tril operator through the [aclnnTril](./docs/aclnnTril&aclnnInplaceTril_en.md) API. |
