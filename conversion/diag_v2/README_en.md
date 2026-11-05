# DiagV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |
| <term>Kirin X90 processor series products</term> |    √     |
| <term>Kirin 9030 processor series products</term> |    √     |

## Function Description

- Operator function: bidirectional diagonal operator (with the same semantics as numpy.diag).

  If the input is a vector (one-dimensional vector), a two-dimensional matrix tensor is returned. The input elements are filled into the diagonal specified by diagonal, and the other positions are filled with 0;

  If the input is a two-dimensional tensor, a one-dimensional vector is output, whose values are the diagonal elements of the input matrix specified by diagonal.

  If diagonal = 0, the main diagonal is selected;

  If diagonal > 0, the diagonal-th diagonal above the main diagonal is selected;

  If diagonal < 0, the -diagonal-th diagonal below the main diagonal is selected.

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
      <td>Input tensor.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BFLOAT16, DOUBLE, BOOL, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the position of the selected diagonal.</li><li>The default value is 0, which indicates the main diagonal.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BFLOAT16, DOUBLE, BOOL, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 and COMPLEX are not supported.

## Constraint Description

- The input is a one-dimensional or two-dimensional tensor (rank 1/2), and the data type of the output is the same as that of the input.
- When the input is 2-dimensional, the value of diagonal must satisfy -(m-1) <= diagonal <= (n-1), where (m,n) is the shape of the input.

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_diag](./examples/test_aclnn_diag.cpp) | Invokes the DiagV2 operator through the [aclnnDiag](./docs/aclnnDiag_en.md) API. |
| GE graph-mode invocation | [test_geir_diag_v2](./examples/test_geir_diag_v2.cpp) | Invokes the DiagV2 operator by building a graph through the [operator IR](./op_graph/diag_v2_proto.h). |
