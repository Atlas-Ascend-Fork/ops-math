# PopulationCount

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: counts the number of bits with bit=1 in the binary representation of each 16-bit integer in the input tensor element by element (population count, also known as Hamming weight / popcount).

- Computation formula:

$$y_i = \text{popcount}(x_i)$$

  where:

- $x_i$: the i-th element of the input tensor, interpreted as an unsigned 16-bit width (negative INT16 values are counted bit by bit in two's complement)

- $y_i$: the i-th element of the output tensor, with the value range $[0, 16]$

- Use cases: scenarios that require counting binary bits, such as Hamming distance computation, sparsity analysis, and bitmask processing.

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>x</td>
    <td>Input</td>
    <td>Integer tensor to be counted, which is x in the formula. 0-8 dimensional tensors are supported, and a 0-dimensional tensor indicates a scalar.</td>
    <td>INT16, UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Bit count result of each element, which is y in the formula. The shape is the same as that of x.</td>
    <td>UINT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- The data type of the input x must be INT16 or UINT16, and the data type of the output y is fixed to UINT8.
- The shape of the output y must be exactly the same as the shape of the input x. Broadcast is not supported.
- 0-8 dimensional tensors are supported, and a 0-dimensional tensor indicates a scalar.
- Empty tensors (with 0 elements) are supported, and an empty result is returned directly.
- Dynamic shape and dynamic rank are supported. The operator can still be correctly executed when the shape is uncertain.
- Both contiguous and non-contiguous inputs are supported. Non-contiguous inputs are automatically converted to contiguous by the framework before computation.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|----------|----------|------|
| Graph-mode invocation | [test\_geir\_population\_count](./examples/test_geir_population_count.cpp) | Invokes the PopulationCount operator by building a graph through the [operator IR](./op_graph/population_count_proto.h). |
