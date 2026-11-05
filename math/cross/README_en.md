# Cross

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √    |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: performs the linear_cross operation on the input tensors.

- Computation formula:

$$
out = self\times other = \begin{vmatrix}i&j&k\\x_1&y_1&z_1\\x_2&y_2&z_2\end{vmatrix} = (y_1z_2-y_2z_1)i-(x_1z_2-x_2z_1)j+(x_1y_2-x_2y_1)k
$$

## Parameter Description

<table class="tg" style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 50px">
  <col style="width: 70px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 50px">
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
      <td>Input self in the computation formula.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, INT16, DOUBLE, INT64, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>Input</td>
      <td>Input other in the computation formula.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, INT16, DOUBLE, INT64, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the computation formula.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8, INT16, DOUBLE, INT64, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products: the supported data types are INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, FLOAT64, COMPLEX64, and COMPLEX128.
- Atlas A2 products and Atlas A3 products: the supported data types are INT8, INT16, INT32, INT64, UINT8, FLOAT16, BFLOAT16, FLOAT, FLOAT64, COMPLEX64, and COMPLEX128.

## Constraint Description

None.

## Invocation Description

| Invocation Method | Invocation Sample                                                    | Description                                                               |
|--------------|---------------------------------------------------------|------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_linalg_cross.cpp](examples/test_aclnn_linalg_cross.cpp) | Invokes the Cross operator through the [aclnnLinalgCross](docs/aclnnLinalgCross_en.md) API. |
| Graph-mode invocation | [test_geir_linalg_cross.cpp](examples/test_geir_linalg_cross.cpp) | Invokes the Cross operator by building a graph through the [operator IR](op_graph/cross_proto.h).             |
