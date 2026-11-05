# OnesLike

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns a tensor with the same shape and type, with all elements set to 1.

- Computation formula:

  $$
    {selfRef}_{i} = 1
  $$

- Example:

  ```text
  Input input:
  tensor([[1, 2],
          [3, 4]])
  Output output:
  tensor([[1, 1],
          [1, 1]])
  ```

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
      <td>Input parameter for the oneslike computation, which is selfRef in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT32, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the oneslike computation. The selfRef tensor is filled with all 1s.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT32, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_inplace_one](./examples/test_aclnn_inplace_one.cpp) | Invokes the OnesLike operator through the [aclnnInplaceOne](./docs/aclnnInplaceOne_en.md) API. |
