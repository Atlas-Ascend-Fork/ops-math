# NanToNum

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: replaces the NaN, positive infinity, and negative infinity values in the input with the values specified by nan, posinf, and neginf, respectively.

- Computation formula:

$$y=\begin{cases} nan & \text{if } x \text{ is NaN} \\ posinf & \text{if } x \text{ is } +\infty \\ neginf & \text{if } x \text{ is } -\infty \\ x & \text{otherwise} \end{cases}$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 320px">
  <col style="width: 290px">
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
      <td>Input parameter for the nan_to_num computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the nan_to_num computation, which is y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nan</td>
      <td>Attribute</td>
      <td>Value that replaces NaN elements in the tensor.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>posinf</td>
      <td>Attribute</td>
      <td>Value that replaces positive infinity elements in the tensor.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>neginf</td>
      <td>Attribute</td>
      <td>Value that replaces negative infinity elements in the tensor.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_nan_to_num](./examples/test_aclnn_nan_to_num.cpp) | Invokes the NanToNum operator through the [aclnnNanToNum](./docs/aclnnNanToNum&aclnnInplaceNanToNum_en.md) API. |
