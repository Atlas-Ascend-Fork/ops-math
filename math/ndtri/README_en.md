# Ndtri

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: computes the inverse standard normal cumulative distribution function (probit / inverse normal CDF). It computes the standard normal quantile element by element for the input probability tensor. The bottom layer uses the piecewise rational approximation algorithm of the Cephes math library.

- Computation formula:

$$y_i = \mathrm{ndtri}(x_i) = \Phi^{-1}(x_i) = \sqrt{2} \cdot \mathrm{erf}^{-1}(2 x_i - 1), \quad x_i \in (0, 1)$$

- Special value handling:

| Input $x_i$ | Output $y_i$ |
| :---: | :---: |
| `0` | `-inf` |
| `1` | `+inf` |
| Positive and negative FP32/BF16 subnormal values | `-inf`; the inputs are treated as zero during computation |
| $x_i < 0$ (excluding the subnormal values above) or $x_i > 1$ | `NaN` |
| `NaN` / `+inf` / `-inf` | `NaN` |

FP16 inputs are first promoted to FP32. FP16 subnormal values become normal FP32 values and are processed by the original numerical algorithm.

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
      <td>Probability tensor for the ndtri computation, which is x_i in the formula. The recommended value range is (0, 1).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Computation result of the inverse standard normal CDF, which is y_i in the formula. The output has the same shape and data type as the input. A scalar input produces a scalar output, and an empty tensor produces an empty tensor of the same shape.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input x has at most 8 dimensions.
- At runtime, each dimension of input x must be a non-negative integer, and the total number of elements must not exceed the maximum value of INT64.

## Invocation Description

| Invocation Method | Sample Code                                       | Description                                                              |
|--------------|--------------------------------------------|-----------------------------------------------------------------|
| GE graph mode | [test_geir_ndtri](./examples/test_geir_ndtri.cpp) | Invokes the Ndtri operator by building a graph through the [operator IR](./op_graph/ndtri_proto.h). |
