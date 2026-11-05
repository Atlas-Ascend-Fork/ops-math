# GroupedBiasAddGrad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs reduction summation on the bias gradients of grouped channels.

- Computation formula:

&emsp;&emsp;(1) When the optional input groupIdxOptional exists and groupIdxType is 0:

$$
out(G,H) =
\begin{cases}
\displaystyle
\sum_{i = \mathrm{groupIdxOptional}(j-1)}^{\mathrm{groupIdxOptional}(j)}  \!\! \mathrm{gradY}(i, H), & 1 \leq j \leq G-1 \\[8pt]
\displaystyle
\sum_{i = 0}^{\mathrm{groupIdxOptional}(j)}  \mathrm{gradY}(i, H), & j = 0
\end{cases}
$$

&emsp;&emsp;(2) When the optional input groupIdxOptional exists and groupIdxType is 1:

$$
groupIdx(i) = \sum_{i=0}^{j} groupIdxOptional(j), j=0...G
$$

$$
out(G,H) =
\left\{
\begin{aligned}
&\sum_{i\,=\,\mathrm{groupIdx}(j-1)}^{\mathrm{groupIdx}(j)} \!\! \mathrm{gradY}(i, H), && 1 \leq j \leq G-1 \\
&\sum_{i\,=\,0}^{\mathrm{groupIdx}(j)} \mathrm{gradY}(i, H), && j = 0
\end{aligned}
\right.
$$

&emsp;&emsp;In the formulas, gradY is 2-dimensional, H indicates the size of the last dimension of gradY, G indicates the size of dimension 0 of groupIdxOptional, that is, groupIdxOptional contains G numbers, and groupIdxOptional(j) indicates the value of the j-th number. After computation, out is 2-dimensional, and its shape is (G, H).<br>
&emsp;&emsp;(3) When the optional input groupIdxOptional does not exist:

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;In the formula, gradY is 3-dimensional, and G, C, and H indicate the sizes of dimensions 0 to 2 of gradY in order. After computation, out is 2-dimensional, and its shape is (G, H).

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
      <td>gradY</td>
      <td>Input</td>
      <td>Gradient for backpropagation, which is input gradY in the formula. A <a href="../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensor</a> is supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIdxOptional</td>
      <td>Optional input</td>
      <td>End position of each group, which is input groupIdxOptional in the formula. A maximum of 2048 groups are supported, and a non-contiguous tensor is supported.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient of bias, which is out in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIdxType</td>
      <td>Optional attribute</td>
      <td>Indicates the type of groupIdx. Supported values:<br>
      0: indicates that the values in groupIdxOptional are the end indexes of groups.<br>
      1: indicates that the values in groupIdxOptional are the sizes of groups.</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- When the input group_idx exists, the following constraints must be satisfied:
  - Ensure that the values of the tensor do not exceed the maximum value of INT32 and are non-negative.
  - grad_y supports only 2-dimensional shapes.
- When the input group_idx does not exist, grad_y supports only 3-dimensional shapes.
- When the input group_idx exists and group_idx_type is 0, ensure that the tensor data is arranged in ascending order and the last value equals the size of dimension 0 of grad_y.
- When the input group_idx exists and group_idx_type is 1, it must be ensured that the sum of the tensor data equals the size of dimension 0 of grad_y.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_bias_add_grad](./examples/test_aclnn_grouped_bias_add_grad.cpp) | Invokes the GroupedBiasAddGrad operator through the [aclnnGroupedBiasAddGrad](./docs/aclnnGroupedBiasAddGrad_en.md) API. |
| Graph-mode invocation | [test_geir_grouped_bias_add_grad](./examples/test_geir_grouped_bias_add_grad.cpp)   | Invokes the GroupedBiasAddGrad operator by building a graph through the [operator IR](./op_graph/grouped_bias_add_grad_proto.h). |
