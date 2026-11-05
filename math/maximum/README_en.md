# Maximum

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term> |√|
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns a new tensor composed of the maximum values of the two input tensors.

- Computation formula:

$$out[i] = max(x_1[i], x_2[i])$$

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
      <td>Input parameter for the Maximum computation, which is x1 in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for the Maximum computation, which is x2 in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Element-wise maximum result of the tensors, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_maximum](./examples/test_aclnn_maximum.cpp) | Invokes the aclnnMaximum operator through the [aclnnMaximum](./docs/aclnnMaximum_en.md) API. |
| aclnn invocation | [test_aclnn_maxn](./examples/test_aclnn_maxn.cpp) | Invokes the aclnnMaxN operator through the [aclnnMaxN](./docs/aclnnMaxN_en.md) API. |
