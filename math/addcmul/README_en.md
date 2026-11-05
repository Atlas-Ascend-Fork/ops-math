# Addcmul

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: a tensor computation function that performs the combined multiply-divide-add operation and merges tensor multiplication (with scaling) + tensor addition into a single operation.

- Computation formula:

$out = self + value × input_1 × input_2$

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
      <td>Input parameter for the addcmul computation, which is self in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_1</td>
      <td>Input</td>
      <td>Input parameter for the addcmul computation, which is input_1 in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_2</td>
      <td>Input</td>
      <td>Input parameter for the addcmul computation, which is input_2 in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input parameter for the addcmul computation, which is value in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the addcmul computation, which is out in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_addcmul](./examples/test_aclnn_addcmul.cpp) | Invokes the Addcmul operator through the [aclnnAddcmul](./docs/aclnnAddcmul&aclnnInplaceAddcmul_en.md) API. |
| Graph-mode invocation |  |  |

This directory contains only the aclnn API of the Addcmul operator. If you want to contribute the Ascend C implementation of this operator, refer to [Contribution Process](../../CONTRIBUTING_en.md).
