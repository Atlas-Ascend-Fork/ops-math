# MaskedSelectV3

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: selects elements from the input tensor (self) based on the values in a Boolean mask tensor (mask) as the output, forming a new one-dimensional tensor.

## Parameter Description

<table style="table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>Input tensor</td>
    <td>Input tensor. Its shape must satisfy the broadcast relationship with mask. Non-contiguous Tensors are supported.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mask</td>
    <td>Input tensor</td>
    <td>Boolean mask tensor. Its shape must satisfy the broadcast relationship with self. Non-contiguous Tensors are supported.</td>
    <td>BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>Output one-dimensional tensor. The number of elements equals the dimension size after mask and self are broadcast. Non-contiguous Tensors are not supported.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

- The shapes of self and mask must be able to perform the broadcast operation.
- The shape of out must be one-dimensional, and the number of elements equals the dimension size after self and mask are broadcast.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_masked_select](./examples/test_aclnn_masked_select.cpp) | Invokes the MaskedSelect operator through the [aclnnMaskedSelect](docs/aclnnMaskedSelect_en.md) API. |
