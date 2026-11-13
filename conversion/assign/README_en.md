# Assign

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>     | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: fills the value of one input tensor into another input tensor.

- Computation formula:

$$out_i=value_i$$

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
      <td>ref</td>
      <td>Input</td>
      <td>Input parameter for the Assign computation, which is out in the formula. The input is also the output.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Fill value for the Assign computation, which is value in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ref</td>
      <td>Output</td>
      <td>Output parameter for the Assign computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, and COMPLEX64.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The supported data types are FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, and BOOL.
- <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, and BOOL.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_assign](./examples/test_geir_assign.cpp)   | Invokes the Assign operator by building a graph through the [operator IR](./op_graph/assign_proto.h). |
