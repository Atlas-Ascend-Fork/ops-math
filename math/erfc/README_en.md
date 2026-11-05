# Erfc

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: returns the value of the complementary error function for each element in the input Tensor.

- Computation formula:

$$y=erfc(x)=1-\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t$$

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
      <td>Input parameter for the erfc computation, which is x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the erfc computation, which is y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas A2 products, Atlas inference products, and Atlas training products: the BFLOAT16 data type is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_erfc](./examples/test_aclnn_erfc.cpp) | Invokes the Erfc operator through the [aclnnErfc](./docs/aclnnErfc&aclnnInplaceErfc_en.md) API. |
| Graph mode | [test_geir_erfc](./examples/test_geir_erfc.cpp) | Invokes the Erfc operator by building a graph through the [operator IR](./op_graph/erfc_proto.h). |
