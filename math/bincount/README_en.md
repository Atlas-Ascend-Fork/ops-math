# Bincount

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the frequency of each number in an array of non-negative integers. minlength is the minimum size of the output tensor. When weights is a null pointer, the frequency increases by 1 each time self[i] in self occurs. When weights is not empty, the frequency increases by weights[i] each time self[i] occurs. The result is finally stored at index self[i] in out. Therefore, the size of out is the maximum of (the maximum value in self + 1) and minlength.

- Computation formula:

  If n is the value of self at position i and weights is specified:

  $$
  out[self_i] = out[self_i] + weights_i
  $$

  Otherwise:

  $$
  out[self_i] = out[self_i] + 1
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
      <td>array</td>
      <td>Input</td>
      <td>Input tensor, which is self_i in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>Input</td>
      <td>Specifies the minimum length of the output tensor.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Weight of the input tensor, which is weights_i in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
     <tr>
      <td>bins</td>
      <td>Output</td>
      <td>Output tensor, which is out_i in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The value of the input size must be greater than or equal to the maximum value of the input array plus 1
- The length of the output bins must be equal to the value of size

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_bincount](./examples/test_aclnn_bincount.cpp) | Invokes the Bincount operator through the [aclnnBincount](./docs/aclnnBincount_en.md) API. |
