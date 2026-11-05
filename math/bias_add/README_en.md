# BiasAdd

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>           |  ×   |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function Description

- Operator function: adds a bias value to each element in the input tensor.

- Computation formula:

$$out_i= x_i + bias_i$$

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 330px">
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
      <td>Input parameter for the BiasAdd computation, which is x_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64</td>
      <td>NCHW, NHWC, NDHWC, NCDHW, ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Accumulated bias, which is bias_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the BiasAdd computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64</td>
      <td>NCHW, NHWC, NDHWC, NCDHW, ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_bias_add](./examples/test_geir_bias_add.cpp)   | Invokes the BiasAdd operator by building a graph through the [operator IR](./op_graph/bias_add_proto.h). |
