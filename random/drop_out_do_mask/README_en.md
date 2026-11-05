# DropOutDoMask

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

- Operator function: during training, randomly sets the elements in the input to zero with the probability keep_prob, and scales up the output by a factor of 1/keep_prob. If the corresponding bit in mask is 1, the corresponding element in the input is scaled up. If the bit in mask is 0, the corresponding element in x is set to zero. Specifically, if keep_prob is 1, the elements of x are not changed; if keep_prob is 0, all elements are set to 0.

- Computation formula:

$$
y_i=\begin{cases}
0,&\text { with probability }1-keep\_prob \\
\frac{1}{keep\_prob}x_i, &\text { with probability }keep\_prob
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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula. Dimensions 0-8 are supported for the shape.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Mask data of the bit type stored in the UINT8 type. The shape must be (align(number of elements of x, 128)/8).</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>Input</td>
      <td>Input keep_prob in the formula, which is the probability value used to compute the scaling ratio of the output data.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula. The data type must be a data type to which x can be converted, and the shape must be the same as that of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

1. The value of keep_prob must be between 0 and 1.
2. The shape of mask must satisfy the following condition: $$\text{mask\_shape} = \frac{\text{align}(\text{num}(x), 128)}{8}$$
3. Dimensions 0-8 are supported.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_drop_out_do_mask](./examples/test_aclnn_drop_out_do_mask.cpp) | Invokes the DropOutDoMask operator through the [aclnnDropoutDoMask](docs/aclnnDropoutDoMask_en.md) API. |
| aclnn invocation | [test_aclnn_drop_out_backward](./examples/test_aclnn_drop_out_backward.cpp) | Invokes the DropoutDoMask operator through the [aclnnDropoutBackward](docs/aclnnDropoutBackward_en.md) API. |
