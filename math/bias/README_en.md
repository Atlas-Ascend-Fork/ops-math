# Bias

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

Bias derives the logical broadcast shape of `bias` according to `axis`, `num_axes`, and `bias_from_blob`, and computes `y = x + broadcast(bias)`.

The computation formula is as follows:

$$y_i = x_i + bias_i$$

The derivation rules of `broadcast(bias)` are as follows:

- If `axis < 0`, execute `axis += rank(x)` first.
- If `bias_from_blob = true` and `num_axes = -1`, `broadcast_shape = [1] * axis + shape(bias)`.
- If `bias_from_blob = true` and `num_axes = 0`, `broadcast_shape = [1] * rank(x)`.
- If `bias_from_blob = true` and `num_axes > 0`, `broadcast_shape = [1] * axis + shape(bias) + [1] * (rank(x) - axis - num_axes)`.
- If `bias_from_blob = false` and `bias` contains a single element, `broadcast_shape = [1] * rank(x)`.
- For other scenarios where `bias_from_blob = false`, `broadcast_shape = [1] * axis + shape(bias) + [1] * (rank(x) - axis - rank(bias))`.

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 380px">
  <col style="width: 250px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor for the Bias computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND, NC1HWC0</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Bias tensor that is broadcast according to the attributes and then added to x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND, NC1HWC0</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Starting alignment axis of the logical shape of bias. The default value is 1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>Attribute</td>
      <td>Number of axes covered by bias. The default value is 1; -1, 0, and positive integers are supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias_from_blob</td>
      <td>Attribute</td>
      <td>Whether to derive the logical broadcast shape according to the blob bias rules. The default value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, whose shape and data type follow those of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND, NC1HWC0</td>
    </tr>
  </tbody></table>

## Constraint Description

- The ND and NC1HWC0 data formats are supported (NC1HWC0 is supported for 4D inputs), as well as inputs and outputs of the same data type among FLOAT, FLOAT16, and BFLOAT16; mixed data types are not supported.
- After attribute-based derivation, each dimension of `bias` must be 1, or the same as the corresponding dimension of `x`.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_bias](./examples/test_geir_bias.cpp) | Invokes the Bias operator by building a graph through the [operator IR](./op_graph/bias_proto.h). |
