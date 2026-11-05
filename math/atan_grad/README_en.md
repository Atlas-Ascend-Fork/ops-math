# AtanGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the input gradient of the arctangent function (atan), which is used for gradient passing in the backpropagation of neural networks.
- Computation formula (the naming is consistent with op_graph/atan_grad_proto.h):

  $$
  z_i = dy_i \times \frac{1}{1 + y_i^2}
  $$

  where $y$ is the input tensor of the forward computation (the independent variable of the atan function), $dy$ is the gradient tensor passed from upstream, and $z$ is the output input-gradient tensor.

- Equivalent step-by-step computation (the NPU implementation uses `Div` instead of `Reciprocal+Mul` to avoid insufficient precision in INTRINSIC mode):
  1. $t_i = y_i \times y_i$ (computes $y^2$)
  2. $g_i = t_i + 1.0$ (computes $1 + y^2$)
  3. $z_i = dy_i / g_i$ (direct division, which is equivalent to $dy_i \times (1 / (1+y^2))$)

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
    <td>The input tensor of the forward computation, which corresponds to y in the formula and is the independent variable of the atan function.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>Input</td>
    <td>The gradient tensor passed from upstream, which corresponds to dy in the formula. The data type must be exactly the same as that of y.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>z</td>
    <td>Output</td>
    <td>The output input-gradient tensor, which corresponds to z in the formula. The data type must be exactly the same as that of y.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- The data types of y, dy, and z must be exactly the same. Implicit type conversion is not supported.
- The shapes of y, dy, and z must be exactly the same. Broadcasting is not supported.
- Supports empty Tensors (with 0 elements).
- Supports Tensors of 0 to 8 dimensions. Dimension 0 indicates a scalar, and in this case dy and z must also be 0-dimensional.
- When the value of y is extremely large (for example, the maximum value of fp16), $y^2$ may overflow to inf. In this case, $1/\text{inf}=0$, z=0, which is normal numerical behavior.
- For FP16/BF16 inputs: the NPU kernel uses the Cast mode of **FP16/BF16 → FP32 intermediate computation → FP16/BF16** (the default implementation convention of CANN kernels). For FP32 inputs: the computation is directly performed in FP32.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_atan_grad.cpp](examples/test_geir_atan_grad.cpp) | Invokes the AtanGrad operator through the graph mode. |
