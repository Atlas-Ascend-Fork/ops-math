# Trace

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- |:----:|
| <term>Ascend 950PR&950DT products</term>                             |  √   |
| <term>Atlas A3 products</term>     |  √   |
| <term>Atlas A2 products</term> |  √   |
| <term>Atlas 200I/500 A2 inference products</term>                      |  ×   |
| <term>Atlas inference products</term>                             |  √   |
| <term>Atlas training products</term>                              |  √   |

## Function Description

- Operator function:
  The Trace operator computes the sum of the main diagonal elements of a matrix, starting from the upper left corner.

- Computation formula:

  $$
  out=sum(diag(self))
  $$

## Parameter Description

  <table style="table-layout: fixed; width: 1464px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 300px">
  <col style="width: 324px">
  <col style="width: 118px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input tensor, which is the input self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, COMPLEX64, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is the output out in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, COMPLEX64, INT64</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

1. `self` must be a 2D Tensor.
2. In graph mode, the output is a 0D scalar.
3. During an ACLNN invocation, `out` must be a 0D Tensor.
4. For a non-square matrix with the shape $(M, N)$, the accumulated diagonal has a length of $\min(M, N)$.
5. A non-2D input, or a non-0D output during an ACLNN invocation, returns a parameter error.
6. An empty 2D Tensor is supported, and the result is the 0D scalar 0.

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| aclnn invocation  | [test_aclnn_trace](./examples/arch35/test_aclnn_trace.cpp) | Invokes the Trace operator through the [aclnnTrace](./docs/aclnnTrace_en.md) API.   |
| Graph-mode invocation | [test_geir_trace](./examples/test_geir_trace.cpp) | Invokes the Trace operator by building a graph through the [operator IR](./op_graph/trace_proto.h) |
