# Logdet

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the natural logarithm of the determinant of the input self.

- Computation formula:

  $$
  out = log(det(self))
  $$

    - If the result of $det(self)$ is 0, then $out = -inf$.
    - If the result of $det(self)$ is a negative number, then $out = nan$.

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
      <td>Input self in the computation formula.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the computation formula.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

Overflow values Inf/Nan in the input data are not supported.

## Invocation Description

| Invocation Method | Invocation Sample                                                            | Description                                               |
|--------------|-----------------------------------------------------------------|--------------------------------------------------|
| aclnn invocation | [test_aclnn_logdet.cpp](examples/test_aclnn_logdet.cpp) | Invokes the Logdet operator through the [aclnnLogdet](docs/aclnnLogdet_en.md) API. |
