# ClipByNormNoDivSum

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs the ClipByNorm variant computation on the elements of four inputs. Finally, normalization by dividing by the norm is not performed; instead, piecewise mapping is implemented through the Select/Max chain. The formula is described as follows:

- Computation formula:

$$y_i = \max(\text{select\_ones}(x_i \leq \text{greater\_zeros}_i, x_i, \text{sqrt}(\text{select\_ones}(x_i > \text{greater\_zeros}_i, x_i, \text{select\_ones}_i))), \text{maximum\_ones}_i)$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 230px">
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
      <td>The first input tensor, which is x in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_zeros</td>
      <td>Input</td>
      <td>The second input tensor, the comparison threshold, which is greater_zeros in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_ones</td>
      <td>Input</td>
      <td>The third input tensor, the alternative value for conditional selection, which is select_ones in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maximum_ones</td>
      <td>Input</td>
      <td>The fourth input tensor, the maximum clipping boundary, which is maximum_ones in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result tensor of the computation, which is y in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The data types of all inputs and outputs must be the same.
- All inputs support the broadcast semantics, and the output shape is the shape after broadcasting the four inputs.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|----------|------|
| Graph-mode invocation | [test_geir_clip_by_norm_no_div_sum](./examples/test_geir_clip_by_norm_no_div_sum.cpp) | Invokes the ClipByNormNoDivSum operator by building a graph through the [operator IR](./op_graph/clip_by_norm_no_div_sum_proto.h). |
