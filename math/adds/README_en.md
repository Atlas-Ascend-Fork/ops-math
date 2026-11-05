# Adds

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: implements the addition of a tensor and a scalar. It adds a scalar value to each element of the input tensor and outputs the result tensor.
- Computation formula:

  $$
  y_i = x_i + scalar
  $$

  where:
  - $x_i$: the i-th element of the input tensor
  - $scalar$: the scalar value
  - $y_i$: the i-th element of the output tensor

- Usage scenario: the scalar addition operation in deep learning models, commonly used in scenarios such as bias addition and constant adjustment.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>The input tensor, which corresponds to x in the formula. Tensors of 1 to 8 dimensions are supported.</td>
    <td>FLOAT16, FLOAT32, BFLOAT16, INT16, INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Attribute</td>
    <td>A required scalar attribute, which corresponds to scalar in the formula.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The output tensor, which corresponds to y in the formula. The data type is the same as that of the input tensor x, and the shape is the same as that of x.</td>
    <td>FLOAT16, FLOAT32, BFLOAT16, INT16, INT32, INT64</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- **Data type constraint**: The data type of the input tensor x must be the same as that of the output tensor y.
- **Shape constraint**: The shape of the output tensor y must be exactly the same as the shape of the input tensor x.
- **Dimension constraint**: The input and output tensors support 1 to 8 dimensions.

## Invocation Description

| Invocation Method | Sample Code  | Description     |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test\_geir\_adds](./examples/test_geir_adds.cpp) | Invokes the Adds operator by building a graph through the [operator IR](./op_graph/adds_proto.h). |
