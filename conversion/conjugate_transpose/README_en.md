# ConjugateTranspose

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: transposes the input x according to the dimension order specified by perm, and then computes the conjugate of the result. For real number types, the conjugate is equivalent to the identity, which is equivalent to a normal Transpose.

- Computation formula: y = conj(transpose(x, perm)), where the size of the i-th dimension of y equals the size of the perm[i]-th dimension of x.

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
      <td>Input tensor to be transposed. The number of dimensions must be greater than 1.</td>
      <td>COMPLEX64, COMPLEX128, FLOAT16, FLOAT, DOUBLE, BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>perm</td>
      <td>Input</td>
      <td>1-D tensor that describes the dimension permutation order. The number of elements equals the number of dimensions of x.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after transposition and conjugation. Its type is the same as x.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- The number of dimensions of x must be greater than 1 and must not exceed 7.
- perm must be 1-D. The number of elements equals the number of dimensions of x, and the values are a permutation of the dimensions of x.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_conjugate_transpose](./examples/test_geir_conjugate_transpose.cpp)   | Invokes the conjugate_transpose operator by building a graph through the [operator IR](./op_graph/conjugate_transpose_proto.h). |
