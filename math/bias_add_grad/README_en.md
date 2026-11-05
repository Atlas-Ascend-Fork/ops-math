# BiasAddGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the gradient of the bias.

- Computation formula:

$$\text{y}[c] = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} \text{x}[n, c, h, w]$$

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
      <td>Input parameter for the BiasAdd computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Accumulated bias, which is y_c in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Format of the input tensor. Optional.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_bias_add](./examples/test_geir_bias_add_grad.cpp)   | Invokes the BiasAddGrad operator by building a graph through the [operator IR](./op_graph/bias_add_grad_proto.h). |
