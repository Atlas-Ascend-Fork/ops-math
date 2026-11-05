# AddN

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

- Operator function: performs element-wise addition on the inputs and computes the sum.

- Computation formula:

  $$
  y = x_1 + x_2+ ... +x_n
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor x in the formula.</td>
      <td>INT32, INT64, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor y in the formula.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>N</td>
      <td>Attribute</td>
      <td>Number of input x tensors in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| Graph-mode invocation | [test_geir_add_n](./examples/test_geir_add_n.cpp)   | Invokes the AddN operator by building a graph through the [operator IR](./op_graph/add_n_proto.h)                                      |
| aclnn invocation | [test_aclnn_add_n](./examples/test_aclnn_add_n.cpp) | Invokes the AddN operator through the [aclnn API](./op_api/aclnn_add_n.h)                    |
