# Complex

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: builds a complex tensor from the real part real and the imaginary part imag.

- Computation formula:

  $$
  \text{out}[i] = \text{real}[i] + \text{imag}[i] * \mathrm{j}
  $$

  where $\mathrm{j}$ is the imaginary unit, and real and imag participate in the computation with the shapes after broadcasting.

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
      <td>real</td>
      <td>Input</td>
      <td>Real part of the complex number, which is real in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>imag</td>
      <td>Input</td>
      <td>Imaginary part of the complex number, which is imag in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Complex tensor, which is out in the formula.</td>
      <td>COMPLEX64, COMPLEX32, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The data types of real and imag must be the same, the dimensions of both cannot exceed 8, and the shapes must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md).
- The shape of out must be the shape of real and imag after broadcasting.
- On <term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>, when the input is FLOAT, FLOAT16, or DOUBLE, the output data type should be COMPLEX64, COMPLEX32, or COMPLEX128 respectively.
- real, imag, and out all support non-contiguous tensors.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_complex](./examples/test_aclnn_complex.cpp) | Invokes the Complex operator through the [aclnnComplex](./docs/aclnnComplex_en.md) API. |
