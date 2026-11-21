# ReduceStd

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>           |  ×   |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function Description

- Operator function: computes the standard deviation of the specified dimension (dim). This dim can be a single dimension, a dimension list, or None. The computation logic is consistent with ReduceStdV2.
- Computation formula:
  Assume that dim is $i$, and the computation is performed on this dimension. $N$ is the shape of this dimension. Take $self_{i}$ and compute the mean value $\bar{x_{i}}$ on this dimension.

  $$
  y1 = \sqrt{\frac{1}{max(0, N - \delta)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  $$

  When `unbiased=true`, `\delta=1` (sample standard deviation, Bessel correction). When `unbiased=false`, `\delta=0` (population standard deviation).

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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor for the ReduceStd computation, which is self in the formula.</td>
      <td>FLOAT, FLOAT16</td>
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
      <td>unbiased</td>
      <td>Input</td>
      <td>Whether to use Bessel correction. When true, the denominator is N-1. When false, the denominator is N.</td>
      <td>BOOL</td>
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
      <td>y1</td>
      <td>Output</td>
      <td>Output parameter of the standard deviation for the ReduceStd computation.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Output parameter of the mean for the ReduceStd computation.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_reduce_std](./examples/test_geir_reduce_std.cpp) | Invokes the ReduceStd operator by building a graph through the [operator IR](./op_graph/reduce_std_proto.h). |
