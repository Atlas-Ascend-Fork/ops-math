# LogAddExp

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT products</term>       |   √  |
| <term>Atlas A3 products</term> |   √  |
| <term>Atlas A2 products</term> |   √  |
| <term>Atlas 200I/500 A2 inference products</term>          |   ×  |
| <term>Atlas inference products</term>                    |   √  |
| <term>Atlas training products</term>                    |   √  |

## Function Description

- Operator function: computes the logarithm of the sum of exponentials of two input tensors element-wise. Under the default attributes, the base is e.
- Computation formula:
  $$
  y = \ln\left(e^{x_1} + e^{x_2}\right)
  $$

## Parameter Description

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 320px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Input tensor x_1 in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor x2 in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>Optional attribute</td>
      <td>Base of the exponential and logarithm operations, with a default value of -1.0, indicating the base e</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Optional attribute</td>
      <td>Scaling coefficient of -abs(x1-x2) in the exponential term, with a default value of 1.0</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Optional attribute</td>
      <td>Offset of the exponential term, with a default value of 0.0</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula</td>
      <td>Same as x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- base must be -1.0 or a value greater than 0.
- The data types of x1, x2, and y must be the same.
- The shapes of x1 and x2 must satisfy the broadcast relationship, and the shape of y is the shape after broadcasting.
- <term>Atlas inference products</term> and <term>Atlas training products</term> do not support the BFLOAT16 data type.

## Invocation Description

| Invocation Method    | Invocation Sample                                              | Description                                                                           |
| ------- | ------------------------------------------------- | ---------------------------------------------------------------------------- |
| aclnn invocation | [test\_aclnn\_log\_add\_exp](./examples/test_aclnn_log_add_exp.cpp) | Invokes the LogAddExp operator through the [aclnnLogAddExp](./docs/aclnnLogAddExp_en.md) API |
| aclnn invocation | [test\_aclnn\_log\_add\_exp2](./examples/test_aclnn_log_add_exp2.cpp) | Invokes the LogAddExp operator through the [aclnnLogAddExp2](./docs/aclnnLogAddExp2_en.md) API |
