# ExpSegsumGrad

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

## Function Description

- Operator function: the backward computation of [Segsum](../segsum/README_en.md).
- Computation formula (taking 5D input as an example):

  1. The input gradOutput (N1,N2,N3,N4,N4) is multiplied by the input gradSelf (the output of the forward computation).

     $$
     out\_mul = gradOutput * gradSelf
     $$
  2. Generate the triangular matrix A of the (N4,N4) bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  3. Fill with 0 the elements in the input $out\_mul$ corresponding to the positions with the value True in matrix A.

     $$
     out\_mul_i=
     \begin{cases}out\_mul_i,\quad A_i==False
     \\0, \quad A_i==True
     \end{cases}
     $$
  4. Reverse the second-to-last dimension of out_mul to generate out_flip.
  5. Perform the cumsum accumulation on the second-to-last dimension of $out\_flip$. $out\_cumsum\_{i}$ is the element at the corresponding position in the output tensor viewed from the dimension perspective (the subscripts of other dimensions remain unchanged, and the subscripts of the current dimension increase in sequence).

     $$
     out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
     $$
  6. Reverse the -2 dimension of $out\_cumsum$ to generate out_flip2.
  7. Generate the triangular matrix B of the (N4,N4) bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  8. Fill with 0 the elements in $out\_flip2$ corresponding to the positions with the value True in matrix B.

     $$
     out\_flip2_i=
     \begin{cases}out\_flip2_i,\quad B_i==False
     \\0, \quad B_i==True
     \end{cases}
     $$
  9. Return gradInput as the sum of each row along the last dimension of out\_flip2.

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
      <td>grad_output</td>
      <td>Input</td>
      <td>The gradient for the backward computation, which corresponds to `gradOutput` in the formula. The input supports 4D or 5D dimensions, and the sizes of the last two dimensions are the same.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_self</td>
      <td>Input</td>
      <td>The output of the forward computation, which corresponds to `gradSelf` in the formula. Its shape and data type are the same as those of the input `grad_output`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>The output of the backward computation, which corresponds to `out` in the formula. The output must have one fewer dimension than the input, and 3D or 4D is supported. When the input `grad_output` is 4D, the output dimension sizes are the same as the first 3 dimensions of `grad_output`; when the input `grad_output` is 5D, the output dimension sizes are the same as the first 4 dimensions of `grad_output`. The data type is the same as that of the input `grad_output`.
      </td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_exp_segsum_backward](examples/test_aclnn_exp_segsum_backward.cpp) | Invokes the ExpSegsumGrad operator through the [aclnnExpSegsumBackward](docs/aclnnExpSegsumBackward_en.md) API. |
