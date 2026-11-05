# Conj

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

- Operator function: returns the conjugate of a complex tensor. It takes the conjugate of each complex element in the input (the real part remains unchanged, and the imaginary part is negated).

- Computation formula: output = conj(input), that is, output = a - bi, where input = a + bi

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>input</td>
      <td>Input</td>
      <td>Input complex tensor.</td>
      <td>COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Conjugate of the input tensor, whose shape is the same as that of input.</td>
      <td>COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraint Description

- None.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_conj](./examples/test_geir_conj.cpp)   | Invokes the conj operator by building a graph through the [operator IR](./op_graph/conj_proto.h). |
