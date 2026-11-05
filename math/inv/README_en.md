# Inv

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the reciprocal of each element in the input tensor element-wise.

- Computation formula:

$$
y_{i}=\frac{1}{x_{i}}
$$

- Floating-point types (FLOAT, FLOAT16, BFLOAT16) compute the true reciprocal according to IEEE-754 (division by zero yields $\pm\text{Inf}$, $\pm\text{Inf}$ yields $\pm0$, and NaN yields NaN).

- INT32 computes the integer reciprocal truncated toward zero. For integers, $1/x$ is non-zero only when $|x|=1$, which is equivalent to a three-value mapping:

$$
y_{i}=
\begin{cases}
+1 & x_{i}=+1 \\
-1 & x_{i}=-1 \\
0  & \text{other (including } x_{i}=0, |x_{i}|\ge 2, \text{INT\_MIN}, \text{INT\_MAX)}
\end{cases}
$$

  Here, $x_{i}=0$ is mapped to $0$ without throwing an exception, and $\text{INT\_MIN}(-2147483648)$ and $\text{INT\_MAX}(2147483647)$ are both mapped to $0$ without overflow.

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
      <td>Input parameter for the inv computation, which is x in the formula. Empty tensors and non-contiguous tensors are supported, and the number of shape dimensions ranges from 0 to 8.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the inv computation, which is y in the formula. The shape and data type are the same as those of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- x and y must have the same shape and dtype, with element-wise computation and no broadcasting.
- Dynamic shape support: dynamic shapes and dynamic ranks are supported (-1 indicates an unknown dimension / -2 indicates an unknown rank). The dimension and rank are specified by the actual input at runtime, and after specification, the number of shape dimensions must range from 0 to 8.
- The precision of the floating-point path follows community standards: FLOAT rtol/atol = 1e-4, FLOAT16 = 1e-3, and BFLOAT16 = 4e-3.
- The INT32 path is an exact match (bitwise, rtol = atol = 0). The three-value mapping is implemented through pure integer comparison and selection, without float32 conversion or negating INT_MIN. Therefore, INT_MIN/INT_MAX do not overflow, and x = 0 does not throw an exception.

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|--------------|--------------|
| GE graph mode | [test_geir_inv](./examples/test_geir_inv.cpp) | Invokes the Inv operator by building a graph through the [operator IR](op_graph/inv_proto.h). |
