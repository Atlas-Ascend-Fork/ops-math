# ReduceStdV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>           |  ×   |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: computes the standard deviation of the specified dimension (dim). This dim can be a single dimension, a dimension list, or None.
- Computation formula:
  Assume that dim is $i$, and the computation is performed on this dimension. $N$ is the shape of this dimension. Take $self_{i}$ and compute the mean value $\bar{x_{i}}$ on this dimension.

  $$
  out = \sqrt{\frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  $$

  When `keepdim = true`, the dimension is retained after the reduction, and the value of this dimension in the output shape is 1. When `keepdim = false`, the dimension is not retained.

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
      <td>self</td>
      <td>Input</td>
      <td>Input tensor for the ReduceStdV2 computation, which is self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Dimension that participates in the Reduce computation, which is i in the formula.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>Input</td>
      <td>Correction value, which is delta in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>Input</td>
      <td>Specifies whether to retain the dimensions of the input tensor in the output tensor.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter of the ReduceStdV2 computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_std.cpp](./examples/test_aclnn_std.cpp) | Invokes the ReduceStdV2 operator through the [aclnnStd](./docs/aclnnStd_en.md) API. |
