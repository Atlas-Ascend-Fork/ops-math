# Segsum

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: performs the segment sum computation. It generates a semi-separable matrix with the diagonal of 1 and the upper triangle of 0.
- Computation formula (taking 4D input as an example):

  1. The input self is expanded from (N1,N2,N3,N4) to (N1,N2,N3,N4,1).
  2. Broadcasting is performed to obtain (N1,N2,N3,N4,N4).
  3. A (N4,N4) triangular matrix A of the bool type is generated, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  4. The elements in the input self corresponding to the positions whose values are True in matrix A are filled with 0.

      $$
      self_i=
      \begin{cases}self_i,\quad A_i==False
      \\0, \quad A_i==True
      \end{cases}
      $$

  5. The cumsum accumulation is performed along the second-to-last dimension of self. For an element from the dimension perspective (the indexes of other dimensions remain unchanged, and the index of the current dimension increases sequentially), $selfTemp\_{i}$ is the element at the corresponding position in the output tensor.

     $$
     selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
     $$

  6. A (N4,N4) triangular matrix B of the bool type is generated, where the upper triangle is True, the lower triangle is False, and the diagonal is False.
  7. The elements in selfTemp corresponding to the positions whose values are True in matrix B are filled with -inf.

      $$
      out_i=
      \begin{cases}selfTemp_i,\quad B_i==False
      \\-inf, \quad B_i==True
      \end{cases}
      $$

  8. The exponential of each element in selfTemp is computed.

      $$
      out_i=e^{selfTemp_i}
      $$

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>Input for the segment sum computation, corresponding to `self` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output after the segment sum computation, corresponding to `out` in the formula. The number of output dimensions must be 1 greater than the number of input dimensions, and 4D or 5D is supported. When the input `x` is 3D, the dimension sizes of the first 3 dimensions of the output are the same as those of `x`, and the dimension size of the last dimension is the same as that of the 3rd dimension. When the input `x` is 4D, the dimension sizes of the first 4 dimensions of the output are the same as those of `x`, and the dimension size of the last dimension is the same as that of the 4th dimension. The data type is the same as the data type of the input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_exp_segsum](examples/test_aclnn_exp_segsum.cpp) | Invokes the Segsum operator through the [aclnnExpSegsum](docs/aclnnExpSegsum_en.md) API. |
