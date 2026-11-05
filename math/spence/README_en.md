# Spence

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

- Operator function: computes the Spence function (dilogarithm) value of each element in the input tensor and outputs the result.

- Computation formula:

$$\text{Spence}(x) = \int_1^x \frac{\ln t}{1 - t} \, dt = \text{Li}_2(1 - x)$$

Special values:

- Outputs NaN when $x < 0$
- Outputs $\pi^2/6 \approx 1.6449$ when $x = 0$
- Outputs $0$ when $x = 1$
- Outputs NaN when $x = +\infty$

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
      <td>Independent variable of the Spence function, corresponding to x in the formula. Elements with x &lt; 0 output NaN.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Computation result of the Spence function, corresponding to Spence(x) in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- FLOAT16 and BFLOAT16 inputs are promoted to FLOAT32 for computation inside the kernel, and the results are converted back to the original data type.
- The number of input dimensions must not be greater than 8; otherwise, the operator reports an error. Scalar inputs (processed as a Tensor with one element), empty tensors, dynamic shapes, and dynamic ranks are supported.
- Non-contiguous inputs are automatically made contiguous by the framework before computation.

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|------------------------------------------------------------------|
| GE graph mode | [test_geir_spence](./examples/test_geir_spence.cpp) | Invokes the Spence operator in GE graph mode. The prototype file that defines the operator IR is [spence_proto.h](./op_graph/spence_proto.h). |
