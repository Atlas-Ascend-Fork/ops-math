# Rint

English | [简体中文](./readme.md)

## Product Support Status

| Product                                                  | Supported |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR&950DT products</term>       | √        |
| <term>Atlas A3 products</term> | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>| ×        |
| <term>Atlas inference products</term>| √        |
| <term>Atlas training products</term>| √        |

## Function Description

- Operator function: obtains the integer closest to the input data. If two integers are equally close, the even integer is obtained.

- Computation formula:

$$
out_{i}=rint(input_{i})
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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the rint computation, which is input_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter of the rint computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Invocation Sample                                                | Description                                                |
| ------------| -------------------------------------------------| -------------------------------------------------------------|
| Graph-mode invocation | [test_geir_rint](./examples/test_geir_rint.cpp) | Invokes the Rint operator by building a graph through the [operator IR](./op_graph/rint_proto.h). |
