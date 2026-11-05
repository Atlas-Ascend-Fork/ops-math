# Sub

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |     √      |
| <term>Atlas A3 products</term>     |    √       |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |   √     |
| <term>Atlas training products</term>                              |   √     |

## Function Description

* Operator function: performs the subtraction computation, and the minuend is scaled by alpha.
* Computation formula:

  $$
  out_{i} = self_{i} - alpha \times other_{i}
  $$

  $$
  selfRef_{i}=selfRef_{i}-alpha \times other_{i}
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
      <td>Input parameter for the sub computation, which is self_i in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input parameter for the sub computation, which is other_i in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input parameter for the sub computation, which is alpha in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the sub computation, which is out_i in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                            | Description                                                           |
|--------------|-------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sub](./examples/test_aclnn_sub.cpp) | Invokes the Sub operator through the [aclnnSub](./docs/aclnnSub&aclnnInplaceSub_en.md) API. |
| Graph-mode invocation | [test_geir_sub](./examples/test_geir_sub.cpp) | Invokes the Sub operator by building a graph through the [operator IR](./op_graph/sub_proto.h). |
