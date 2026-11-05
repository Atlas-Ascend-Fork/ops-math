# CumulativeLogsumexp

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: computes the cumulative log-sum-exp of the input tensor `x` along the dimension specified by `axis` and saves the result to the output tensor `y`.
- Computation formula: let $i$ be the subscript on the `axis` dimension, and $S_i$ be the set of prefix or suffix subscripts involved in computing the $i$-th output element, then:

$$
y_i = \log \sum_{j \in S_i} e^{x_j}
$$

When `exclusive` is `false`, $S_i$ includes the current position; when `exclusive` is `true`, $S_i$ does not include the current position. When `reverse` is `false`, accumulation proceeds in the direction of increasing subscripts; when `reverse` is `true`, accumulation proceeds in the direction of decreasing subscripts.

## Parameter Description

<table>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input Tensor. The shape supports 1 to 6 dimensions, and the length of the dimension specified by axis must be greater than 0.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>The dimension along which the cumulative log-sum-exp is computed. It must be a compile-time constant scalar, and the value range is [-rank(x), rank(x)-1].</td>
      <td>INT32, INT64, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>Attribute</td>
      <td>The default value is false. false indicates that the current position is included; true indicates that the current position is not included.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reverse</td>
      <td>Attribute</td>
      <td>The default value is false. false indicates forward accumulation; true indicates backward accumulation.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output Tensor, whose shape is the same as that of x.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- `axis` must be a constant scalar.
- `x` does not support scalar input. A parameter error is returned when the length of the dimension specified by `axis` is 0.
- The shape and data type of `y` must be the same as those of `x`.

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_cumulative_logsumexp.cpp](./examples/test_geir_cumulative_logsumexp.cpp) | Invokes the CumulativeLogsumexp operator by building a graph through the operator IR. |
