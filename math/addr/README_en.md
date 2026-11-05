# Addr

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: computes the outer product of the one-dimensional vectors vec1 and vec2 to obtain a two-dimensional matrix, then multiplies the outer product result matrix by a coefficient, adds it to self multiplied by a coefficient, and outputs the result

- Computation formula:
  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
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
      <td>x1</td>
      <td>Input</td>
      <td>Input parameter for the addr computation, which is self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for the addr computation, which is vec1 in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Input parameter for the addr computation, which is vec2 in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Input parameter for the addr computation, which is β in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input parameter for the addr computation, which is α in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the addr computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - Atlas training products: The supported data types are FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, and BOOL.
  - Atlas A2 products, Atlas A3 products, and Ascend 950PR&950DT products: The supported data types are FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, and BFLOAT16.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_addr](./examples/test_aclnn_addr.cpp) | Invokes the Addr operator through the [aclnnAddr](docs/aclnnAddr&aclnnInplaceAddr_en.md) API. |
