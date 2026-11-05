# DropOutDoMaskV3D

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: during training, randomly keeps the elements in the input with the probability keep_prob, and scales up the output by a factor of 1/keep_prob. If the corresponding bit in mask is 1, the corresponding element in the input is scaled up. If the bit in mask is 0, the corresponding element in gradOutput is set to zero. Specifically, if keep_prob is 0, all elements are set to 0; if keep_prob is 1, the elements of gradOutput are not changed.

- Computation formula:

$$
gradOutput_i=\begin{cases}
0,&mask_i=0 \\
\frac{1}{keep\_prob}self_i, &mask_i=1
\end{cases}
$$

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
      <td>Input self in the formula. Dimensions 0-8 are supported for the shape.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Mask data of the bit type stored in the UINT8 type. The shape must be (align(number of elements of self, 128)/8).</td>
      <td>UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>Attribute</td>
      <td>Probability value used to compute the scaling ratio of the output data.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>gradOutput in the formula. The data type must be a data type to which self can be converted, and the shape must be the same as that of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

1. The value of keep_prob must be between 0 and 1.
2. The shape of mask must satisfy the following condition: $$\text{mask\_shape} = \frac{\text{align}(\text{num}(self), 128)}{8}$$
3. Dimensions 0-8 are supported.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_drop_out_do_mask_v3_d](./examples/test_geir_drop_out_do_mask_v3_d.cpp)   | Invokes the DropOutDoMaskV3D operator by building a graph through the [operator IR](./op_graph/drop_out_do_mask_v3_d_proto.h). |
