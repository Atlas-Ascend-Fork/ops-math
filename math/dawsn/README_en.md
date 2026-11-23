# Dawsn

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

- Operator function: computes the Dawson integral $F(x)$ element-wise.

- Computation formula:

$$F(x) = e^{-x^{2}} \int_{0}^{x} e^{t^{2}} \mathrm{d}t$$

## Parameter Description

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
      <td>Input tensor for the Dawson integral computation, i.e. x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor for the Dawson integral computation, i.e. F(x) in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The rank of x ranges from 0 to 8; 0-d scalar input is supported, and out-of-range inputs are rejected by the framework layer.
- Only the ND storage format is supported; other formats are rejected during Host-side tiling validation.
- Empty tensors (0 elements) are supported; the output is an empty tensor with the same shape.
- The total number of input elements must not exceed 288230376151710720 (approximately $2.88\times10^{17}$); otherwise the operator reports an error and rejects the request.
- Dynamic shape (-1 unknown dimension) and dynamic rank (-2 unknown rank) are supported.

## Invocation Description

| Invocation Method | Sample Code | Description |
|---------|----------------------------------------------------|------|
| GE graph mode | [test_geir_dawsn](./examples/test_geir_dawsn.cpp) | Invokes the Dawsn operator in GE graph mode. |
