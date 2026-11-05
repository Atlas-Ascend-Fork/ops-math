# Expint

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

- Operator function: computes the exponential integral Ei(x) = PV ∫_{-∞}^{x} (eᵗ/t) dt (Cauchy principal value integral). The computation formula is:

$$Ei(x) = PV \int_{-\infty}^{x} \frac{e^{t}}{t} \mathrm{d}t$$

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
      <td>Input parameter for the exponential integral computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the exponential integral computation, which is Ei(x) in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The DOUBLE (FP64) data type is not supported.
- When the input x < 0, the output is NaN; when x = 0, the output is -inf; and when x = +inf, the output is +inf.
- The shape and data type of the output are inferred by the framework based on the input (the computation is element-wise, and the output is the same as the input x), and the data format is ND.
- 0- to 8-dimensional ND tensors are supported, including scalars, empty tensors, dynamic shapes, and dynamic ranks. Tensors with 9 or more dimensions are not supported.
- A non-contiguous tensor input is automatically converted to a contiguous tensor by the framework before execution.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation (static shape) | [test_geir_expint](./examples/arch35/test_geir_expint.cpp) | Invokes the Expint operator through the graph mode. |
| Graph-mode invocation (dynamic shape) | [test_geir_expint_dynamic](./examples/arch35/test_geir_expint_dynamic.cpp) | Invokes the Expint operator through the graph mode, supporting scenarios with unknown dimensions (-1) and unknown ranks (-2). |
