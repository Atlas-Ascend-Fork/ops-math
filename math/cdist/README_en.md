# Cdist

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: computes the p-norm distance between each pair of vectors in two sets of vectors.
- Computation formula: For the input tensors x1 and x2, their shapes are $[B, P, M]$ and $[B, R, M]$ respectively, and the shape of the output tensor y is $[B, P, R]$.

  When $p>0$:

  $$
  y_{b,i,j}=\left(\sum_{k=1}^{M}|x1_{b,i,k}-x2_{b,j,k}|^p\right)^{\frac{1}{p}}
  $$

  When $p=0$, the output is the number of unequal elements in the corresponding vectors; when $p=+\infty$, the output is the maximum of the absolute differences between the elements of the corresponding vectors.

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
      <td>x1</td>
      <td>Input</td>
      <td>The first input tensor, whose shape is [B, P, M].</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input tensor, whose shape is [B, R, M].</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>p-norm distance computation result, whose shape is [B, P, R].</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Optional attribute</td>
      <td>Norm order. The value range is [0, +∞], and the default value is 2.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The data types of x1, x2, and y must be the same.
- The dimensions of x1 and x2 range from 2 to 8, and the lengths of the last dimensions must be the same.
- The dimensions of x1 and x2 except the last two dimensions must satisfy the broadcast relationship; the corresponding output dimensions are the broadcast shape.
- If the second-to-last dimension of x1 is P and the second-to-last dimension of x2 is R, the last two dimensions of the output are [P, R].
- p must be greater than or equal to 0. +∞ is supported, and common values are 0, 1.0, 2.0, and +∞.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_cdist](./examples/test_aclnn_cdist.cpp) | Invokes the Cdist operator through the [aclnnCdist](./docs/aclnnCdist_en.md) API. |
