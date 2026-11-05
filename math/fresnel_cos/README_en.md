# FresnelCos

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

- Operator function: computes the Fresnel cosine integral C(x) = integral_0^x cos(pi*t^2/2) dt.

- Computation formula:

$$C(x) = \int_{0}^{x} \cos\left(\frac{\pi}{2} t^{2}\right) \mathrm{d}t$$

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
      <td>Input parameter for the Fresnel cosine integral computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the Fresnel cosine integral computation, which is C(x) in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- Only the BFLOAT16, FLOAT16, and FLOAT data types are supported.
- The number of dimensions (rank) of x ranges from 0 to 8, and 0-dimensional scalar inputs are supported.
- Empty tensors (0 elements) are supported as inputs: the Tiling on the host side short-circuits and returns directly, and the output is an empty tensor with the same shape (handled automatically).
- Non-contiguous tensor inputs are supported (handled automatically on the host side).
- Dynamic shape support: dynamic shapes and dynamic ranks are supported (-1 for an unknown dimension / -2 for an unknown rank).

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_fresnel_cos](./examples/test_geir_fresnel_cos.cpp) | Invokes the FresnelCos operator through the graph mode. |
