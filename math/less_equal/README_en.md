# LessEqual

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

- Operator function: determines whether the element values in the input self are less than or equal to the value of other, and writes the comparison result of each element of self with the value of other into out.
- Computation formula:

  $$
  out_i = (self_i <= other) ? [True] : [False]
  $$

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>Input tensor self_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, UINT64, UINT8, INT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input tensor other in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, UINT64, UINT8, INT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor out_i in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_le_scalar](./examples/test_aclnn_le_scalar.cpp) | Invokes the LessEqual operator through the [aclnnLeScalar&aclnnInplaceLeScalar](./docs/aclnnLeScalar&aclnnInplaceLeScalar_en.md) API.    |
