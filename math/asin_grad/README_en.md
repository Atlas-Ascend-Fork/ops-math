# AsinGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: AsinGrad is the gradient operator of the Asin (arcsine) operator and is used in the backward propagation phase of automatic differentiation in deep learning frameworks. Given the forward input y and the upstream gradient dy, it computes the output gradient z.
- Computation formula:

  $$
  z_i = \frac{dy_i}{\sqrt{1 - y_i^2}}
  $$

  where:
  - y: the input tensor of the forward Asin operator, with a value range of [-1, 1]
  - dy: the gradient (grad_output) propagated from upstream
  - z: the computed input gradient (grad_input)

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>Input</td>
    <td>The input tensor of the forward Asin operator, which corresponds to y in the formula. Supports tensors of 0 to 8 dimensions and empty tensors. The shape must be exactly the same as the shape of dy.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>Input</td>
    <td>The gradient tensor propagated from upstream, which corresponds to dy in the formula. Supports tensors of 0 to 8 dimensions and empty tensors. The shape must be exactly the same as the shape of y.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>z</td>
    <td>Output</td>
    <td>The computed input gradient tensor, which corresponds to z in the formula. The shape is the same as that of the input, and the data type is the same as that of the input.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- The data types of y and dy must be the same. The data type of z is the same as that of y/dy, and only FLOAT, FLOAT16, and BFLOAT16 are supported.
- The shapes of y and dy must be exactly the same. Broadcasting is not supported.
- Supports tensors of 0 to 8 dimensions and empty tensors (0 elements).
- Only the ND data format is supported. The inputs and outputs are configured with automatic contiguity (AutoContiguous), and non-contiguous tensors are converted into contiguous tensors by the framework before entering the Kernel.
- The value range of y should be within [-1, 1]. When |y| = 1, the computation result is inf; when |y| > 1, the computation result is NaN.
- FLOAT16 and BFLOAT16 are both converted to FLOAT32 for intermediate computation, and the computation results are then converted back to the input data type; FLOAT32 is directly computed in FLOAT32.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_asin_grad.cpp](examples/test_geir_asin_grad.cpp) | Invokes the AsinGrad operator through the GE graph mode. For the operator IR definition, see [asin_grad_proto.h](op_graph/asin_grad_proto.h). |
