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
  y = sum(diag(x))
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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Two-dimensional input tensor whose diagonal elements are summed, corresponding to x in the formula.</td>
      <td>COMPLEX64, FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Sum of the diagonal elements, corresponding to y in the formula. A UINT64 input produces a UINT64 output. COMPLEX64, FLOAT, FLOAT16, and BFLOAT16 inputs preserve their data type. All other inputs produce an INT64 output.</td>
      <td>COMPLEX64, FLOAT, FLOAT16, BFLOAT16, INT64, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas training products</term> and <term>Atlas inference products</term>: The data types of x and y do not support BFLOAT16.

## Constraint Description

- x must be a two-dimensional Tensor, and y is a zero-dimensional scalar.
- For a non-square matrix with shape $(M, N)$, the accumulated diagonal has a length of $\min(M, N)$.
- Empty Tensors with shape $(0, N)$, $(M, 0)$, or $(0, 0)$ are supported and produce 0.

## Invocation Description

| Invocation Method   | Sample Code | Description  |
| ------------ | ------------ | ------------ |
| aclnn API  | [test_aclnn_trace](./examples/arch35/test_aclnn_trace.cpp) | Invokes the Trace operator through the [aclnnTrace](./docs/aclnnTrace_en.md) API.   |
| GE graph mode | [test_geir_trace](./examples/test_geir_trace.cpp) | Invokes the Trace operator by building a graph through the [operator IR](./op_graph/trace_proto.h). |
