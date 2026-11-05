# Polar

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |

## Function Description

- Operator function: builds a complex tensor whose elements are the Cartesian coordinates corresponding to the polar coordinates, where the absolute value is abs and the angle is angle.

- Computation formula:

  $$
  out = abs*cos(angle) + abs*sin(angle)*j
  $$

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
      <td>abs</td>
      <td>Input</td>
      <td>Absolute value, which is abs in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>angle</td>
      <td>Input</td>
      <td>Angle, which is angle in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_polar](./examples/test_aclnn_polar.cpp) | Invokes the Polar operator through the [aclnnPolar](./docs/aclnnPolar_en.md) API. |
