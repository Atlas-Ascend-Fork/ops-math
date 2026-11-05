# MatrixDiag

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

- Operator function: returns a batched diagonal Tensor with the given batched diagonal values.

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
      <td>Input parameter for the MatrixDiag computation, which is x in the formula.</td>
      <td>DOUBLE, FLOAT32, FLOAT16, BFLOAT16, COMPLEX32, COMPLEX64, COMPLEX128, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, QINT8, QUINT8, QINT16, QUINT16, QINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, BOOL.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the MatrixDiag computation, which is out in the formula.</td>
      <td>DOUBLE, FLOAT32, FLOAT16, BFLOAT16, COMPLEX32, COMPLEX64, COMPLEX128, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, QINT8, QUINT8, QINT16, QUINT16, QINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, BOOL.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_matrix_diag](./examples/test_geir_matrix_diag.cpp)   | Invokes the MatrixDiag operator by building a graph through the [operator IR](./op_graph/matrix_diag_proto.h).
