# Rfft1D

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

- Operator function: performs the RFFT (Fourier transform) computation on the input tensor. The output is a complex tensor that contains non-negative frequencies.
- Computation formula:

  $$
  y = W \cdot x
  $$

  Here, W is the DFT matrix, which is defined as $W_{jk}=e^{-ij2\pi{\tfrac{k}{N}}n}$

- Example:

  Assume that x is {1, 2, 3, 4}. Then y = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j}

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>Input tensor x in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dft</td>
      <td>Input</td>
      <td>The input DFT matrix or twiddle matrix. With this matrix, the Rfft1D transform can be expressed as matrix multiplication.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n</td>
      <td>Input</td>
      <td>Signal length. When the value range of n is [1, 4096], the DFT matrix should be passed in for dft. Otherwise, the twiddle matrix should be passed in.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>norm</td>
      <td>Input</td>
      <td>Normalization mode. Supported values: 1 indicates no normalization, 2 indicates normalization by 1/n, and 3 indicates normalization by 1/sqrt(n).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_acl_rfft1_d](./examples/test_acl_rfft1d.cpp) | Invokes the Rfft1D operator through the [aclRfft1D](./docs/aclRfft1D_en.md) API.    |
