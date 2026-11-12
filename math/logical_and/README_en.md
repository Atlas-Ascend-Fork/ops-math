# LogicalAnd

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

- Operator function: performs a logical AND check on the corresponding elements of the two input tensors and outputs a Boolean tensor (True/False).

- Computation formula:

$out=input1∧input2$

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
      <td>input_1</td>
      <td>Input</td>
      <td>Input parameter for the logical_and computation, which is input_1 in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_2</td>
      <td>Input</td>
      <td>Input parameter for the logical_and computation, which is input_2 in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the logical_and computation, which is out in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_logical_and](./examples/test_aclnn_logical_and.cpp) | Invokes the LogicalAnd operator through the [aclnnLogicalAnd](./docs/aclnnLogicalAnd&aclnnInplaceLogicalAnd_en.md) API. |
| Graph-mode invocation | - | Invokes the LogicalAnd operator by building a graph through the [operator IR](./op_graph/logical_and_proto.h). |
