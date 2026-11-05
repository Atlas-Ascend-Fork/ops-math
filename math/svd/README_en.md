# Svd

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- API function: computes the singular value decomposition of one or more matrices.

  When the dimension of the input tensor is greater than 2, the high-dimensional tensor is treated as a batch of matrices. For an input tensor with the shape (..., M, N), the dimensions (...) before the second-to-last dimension are regarded as batch dimensions, and the singular value decomposition is computed independently for each (M, N) matrix.

- Computation formula:

$$
\mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
$$

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
      <td>input</td>
      <td>Input</td>
      <td>Tensor on which the singular value decomposition is to be performed, corresponding to input in the formula.<br>Empty tensors are not supported. The shape has at least 2 dimensions.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Output</td>
      <td>Output tensor, corresponding to sigma in the formula.<br>The shape needs to be deduced based on the shape of input and the fullMatrices parameter. The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>U</td>
      <td>Output</td>
      <td>Output tensor, corresponding to U in the formula.<br>The shape needs to be deduced based on the shape of input and the fullMatrices parameter. The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>V</td>
      <td>Output</td>
      <td>Output tensor, corresponding to V in the formula.<br>The shape needs to be deduced based on the shape of input and the fullMatrices parameter. The data type must be the same as that of input.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>compute_uv</td>
      <td>Attribute</td>
      <td>Optional attribute. The default value is true. Denotes whether to compute the output tensors u and v. If set to true, the output tensors sigma, u, and v are all computed. If set to false, only sigma is computed.</td>
      <td>BOOL</td>
      <td>true/false</td>
    </tr>
     <tr>
      <td>full_matrices</td>
      <td>Attribute</td>
      <td>Optional attribute. Denotes whether to compute the output tensors u and v completely. The default value is false.</td>
      <td>BOOL</td>
      <td>true/false</td>
    </tr>
  </tbody></table>

## Constraint Description

  None

## Invocation Description

  | Invocation Method | Invocation Sample                                                          | Description                                                          |
  |--------|---------------------------------------------------------------|-------------------------------------------------------------|
  | aclnn invocation | [test_aclnn_svd](./examples/test_aclnn_svd.cpp) | Invokes the Svd operator through the [aclnnSvd](./docs/aclnnSvd_en.md) API.|
