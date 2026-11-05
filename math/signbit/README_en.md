# Signbit

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

- Operator function: checks whether the sign bit of each element in the input tensor is 1, that is, whether the element is a negative number or -0.0.
- Computation formula:

  $$
  y_i = (x_i < 0) \: ? \: True : False \quad (returns True when x_i is -0.0)
  $$

- Example:
  - If x=[0, -3, 5], the result of signbit(x) is [False, True, False].
  - If x=[[-0.0, 3.14], [-2.718, inf]], the result of signbit(x) is [[True, False], [True, False]].

## Parameter Description

<table style="table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the sign bit check, which is x_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the sign bit check, which is y_i in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_signbit](./examples/test_aclnn_signbit.cpp) | Invokes the Signbit operator through the [aclnnSignbit](./docs/aclnnSignbit_en.md) API. |
