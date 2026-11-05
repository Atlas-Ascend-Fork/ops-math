# Sign

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |  ×     |

## Function Description

- Operator function: performs the Sign function computation on the input tensor element-wise and outputs the result tensor.

- Computation formula:

  $$
  resultput_i = \left\{
  \begin{aligned}
  1,\quad input_i > 0\\
  0,\quad input_i = 0\\
  -1,\quad input_i < 0
  \end{aligned}
  \right.
  $$

- Computation formula (BOOL type):

  $$
  resultput_i = \left\{
  \begin{aligned}
  \text{True},\quad input_i = \text{True}\\
  \text{False},\quad input_i = \text{False}\\
  \end{aligned}
  \right.
  $$

- Computation formula (complex number, where real and imag below indicate taking the real part and the imaginary part respectively):

  $$
  resultput_i = \alpha \cdot cos(\theta_i) + \beta \cdot sin(\theta_i) \\
  \alpha = \left\{
  \begin{aligned}
  1,\quad real(input_i) > 0 \\
  0,\quad real(input_i) = 0 \\
  -1,\quad real(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \beta = \left\{
  \begin{aligned}
  1,\quad imag(input_i) > 0 \\
  0,\quad imag(input_i) = 0 \\
  -1,\quad imag(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \theta_i = arctan(\frac{imag(input_i)}{real(input_i)})
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
      <td>Input parameter for the sign computation, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, DOUBLE, INT8, INT16, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the sign computation, which is resultput_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, DOUBLE, INT8, INT16, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The INT8, INT16, UINT8, UINT16, UINT32, and UINT64 types are supported only through the geir invocation.
- The BOOL, DOUBLE, COMPLEX64, and COMPLEX128 types are supported only through the aclnn invocation.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sign](./examples/test_aclnn_sign.cpp) | Invokes the Sign operator through the [aclnnSign](./docs/aclnnSign_en.md) API. |
