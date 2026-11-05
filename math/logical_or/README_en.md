# LogicalOr

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

## Function Description

Operator function: performs a logical OR operation on the elements of the given input tensors. When the two input tensors are not of the bool type, 0 is regarded as False and a non-zero value is regarded as True.

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
      <td>Input parameter for the logical_or computation. The shape must satisfy the <a href="../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</td>
      <td>BOOL, INT8, UINT8, INT16, INT32, INT64, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input parameter for the logical_or computation. The shape must satisfy the <a href="../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
      <td>BOOL, INT8, UINT8, INT16, INT32, INT64, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the logical_or computation. The shape is the same as the shape of self and other after broadcasting.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>: All data types (BOOL, INT8, UINT8, INT16, INT32, INT64, FLOAT16, BFLOAT16, FLOAT) are supported.

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - Only the BOOL data type is supported for self and other.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------|
| aclnn invocation | [test_aclnn_logical_or](./examples/test_aclnn_logical_or.cpp) | Invokes the LogicalOr operator through the [aclnnLogicalOr](./docs/aclnnLogicalOr&aclnnInplaceLogicalOr_en.md) API. |
| aclnn invocation | [test_aclnn_inplace_logical_or](./examples/test_aclnn_inplace_logical_or.cpp) | Invokes the LogicalOr operator through the [aclnnInplaceLogicalOr](./docs/aclnnLogicalOr&aclnnInplaceLogicalOr_en.md) API.       |
