# Slogdet

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |     ×     |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             |   √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: computes the sign and natural logarithm of the determinant of the input self.

- Computation formula:

  $$
  signOut = sign(det(self))     \\
  logOut = log(abs(det(self)))
  $$

  Here, det denotes the determinant computation and abs denotes the absolute value computation. If the result of $det(self)$ is 0, $logOut = -inf$.

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>The input self in the computation formula.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logOut</td>
      <td>Output</td>
      <td>The output logOut in the computation formula.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>signOut</td>
      <td>Output</td>
      <td>The output signOut in the computation formula.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- When `self` is of the COMPLEX type, `signOut` cannot be of a non-COMPLEX type.
- When `self` is of the COMPLEX type, `logOut` cannot be of a non-COMPLEX type.

## Constraint Description

Overflow values Inf/Nan are not supported in the input data.

## Invocation Description

| Invocation Method | Invocation Sample                                                      | Description                                                    |
|--------------|-----------------------------------------------------------|-------------------------------------------------------|
| aclnn invocation | [test_aclnn_slogdet.cpp](examples/test_aclnn_slogdet.cpp) | Invokes the Slogdet operator through the [aclnnSlogdet](docs/aclnnSlogdet_en.md) API. |
