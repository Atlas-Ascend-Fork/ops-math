# Tanh

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |     √     |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             |   ×     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

- Operator function: activation function. Returns a tensor with the same shape as the input tensor, and performs the elementwise computation on the input tensor.
- Computation formula:

  $$
  tanh(self)=\frac{e^{self}-e^{-self}}{e^{self} + e^{-self}}
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
      <td>Input parameter for the tanh computation, which is self_i in the formula.</td>
      <td>FLOAT, FLOAT16, BOOL, UINT8, INT8, INT16, INT32, INT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the tanh computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                            | Description                                                           |
|--------------|-------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_tanh](./examples/test_aclnn_tanh.cpp) | Invokes the Tanh operator through the [test_aclnn_tanh](./docs/aclnnTanh&aclnnInplaceTanh_en.md) API. |
