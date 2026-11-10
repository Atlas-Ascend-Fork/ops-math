# Hypot

## Applicable Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                        |    ×     |
| <term>Atlas A3 products</term>                                 |    √     |
| <term>Atlas A2 products</term>                                 |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                |    ×     |
| <term>Atlas inference products</term>                          |    ×     |
| <term>Atlas training products</term>                           |    ×     |

## Description

- Operator function: The operator computes the hypotenuse length (two-dimensional Euclidean norm) elementwise from the corresponding elements of two input tensors. The output is always nonnegative.
- Formula:
  $$
  out_i = \sqrt{self_i^{2} + other_i^{2}}
  $$
- Numerical stability: The implementation scales inputs by the larger absolute value to avoid overflow or underflow in intermediate squared values. Two zero inputs produce `+0`; `inf` takes precedence over `NaN`.

## Parameters

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input tensor self in the formula.</td>
      <td>FLOAT, FLOAT16, and BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input tensor other in the formula.</td>
      <td>FLOAT, FLOAT16, and BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor out in the formula.</td>
      <td>FLOAT, FLOAT16, and BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- self and other can have different dtypes. Type promotion determines the output dtype. The output dtype is FLOAT if either input is FLOAT or if one input is FLOAT16 and the other is BFLOAT16. If both inputs have the same dtype, the output has that dtype.

## Constraints

- The operator supports ND tensors with 0 to 8 dimensions. The input shapes must support broadcasting.
- The operator supports empty tensors and non-contiguous tensors.
- The operator does not support DOUBLE, integer types, BOOL, or complex types.
- This release provides only `aclnnHypot` and the graph-mode `Hypot` operator. It does not provide an in-place API.

## Invocation

| Invocation Method | Sample                                                                 | Description                                                    |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_hypot](./examples/test_aclnn_hypot.cpp) | Invoke the Hypot operator using the [aclnnHypot](./docs/aclnnHypot.md) API. |
| Graph-mode invocation | [test_geir_hypot](./examples/test_geir_hypot.cpp) | Invoke the Hypot operator by constructing a graph using the [operator IR](./op_graph/hypot_proto.h). |
