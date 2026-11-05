# AssignSub

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

- Operator function: completes the subtraction computation on the original tensor.

- Computation formula:

$$out_i = var_i - value_i$$

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
      <td>var</td>
      <td>Input</td>
      <td>The input tensor, which is var_i in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>value</td>
      <td>Input</td>
      <td>The input tensor, which indicates the value to be subtracted from var_i and is out_i in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>The output tensor, which is out_i in the formula and shares the same address as var_i.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Optional. Specifies whether to use a lock to protect the update operation.</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                | Description                                                             |
|--------------|-----------------------------------------------------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_assign_sub.cpp](examples/test_geir_assign_sub.cpp)   | Invokes the AssignSub operator by building a graph through the [operator IR](op_graph/assign_sub_proto.h).                   |
