# DiagPart

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: obtains the diagonal elements of the input tensor.

- Computation formula:

$out= diag(x)$

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
      <td>Input parameter for the DiagPart computation, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the DiagPart computation, which is out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

- The input must have an even number of dimensions, and the corresponding dimensions in the two halves must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_diag_part](./examples/test_geir_diag_part.cpp)   | Invokes the DiagPart operator by building a graph through the [operator IR](./op_graph/diag_part_proto.h). |
