# AcoshGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √      |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: computes the backward gradient of the Acosh (inverse hyperbolic cosine) operator.
- Operator formula:

  $$
  z_i = dy_i \cdot \dfrac{1}{\sqrt{y_i^2 - 1}}
  $$

  where:
  - $y_i$ is the input tensor of the forward Acosh operator, and its value range is expected to be $\geq 1$;
  - $dy_i$ is the gradient passed from upstream;
  - $z_i$ is the gradient with respect to the original input tensor, which equals the upstream gradient multiplied by $1/\sqrt{y_i^2 - 1}$.

- Boundary ($y_i \leq 1$): when $y_i = 1$, $\sqrt{0}=0$, and division by zero results in $+\text{Inf}$; when $y_i < 1$, the square root of a negative number results in NaN, which propagates naturally according to IEEE 754. It is recommended that the value range of the input $y$ fall within $[1, +\infty)$ to avoid Inf or NaN results.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>y</td>
      <td>Input</td>
      <td>The input tensor of the forward Acosh operator. The value range is expected to fall within [1, +∞).</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>The gradient tensor passed from upstream, whose shape and dtype are the same as those of y.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>The gradient with respect to the original input tensor, whose shape and dtype are the same as those of y.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The shapes of y and dy must be exactly the same.
- The dtypes of y and dy must be exactly the same.
- Only the ND format is supported.
- Dynamic shape is supported: the shape is determined at runtime, and dynamic shape and dynamic rank (rank range 0 to 8) are supported. The shapes of y/dy/z must be exactly the same.
- For an empty tensor (with 0 elements), an empty result is returned directly.
- [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the number of dimensions of a non-contiguous tensor cannot be greater than 8.

## Invocation Description

| Invocation Method | Invocation Sample                                                                      | Description                                                                                       |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_acosh_grad](./examples/test_geir_acosh_grad.cpp)             | Invokes the AcoshGrad operator by building a graph through the [operator IR](./op_graph/acosh_grad_proto.h).                 |
