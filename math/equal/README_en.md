# Equal

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |   ×    |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √    |

## Function Description

- Operator function: compares two input tensors element by element for equality and returns a bool tensor.
- Computation expression:

  $$
  out = (self == other)  ?  True : False
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
      <td>x1</td>
      <td>Input</td>
      <td>The input self in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, DOUBLE, INT16, INT64, COMPLEX64, COMPLEX128, QUINT8, QINT8, QINT32, STRING, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The input other in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, DOUBLE, INT16, INT64, COMPLEX64, COMPLEX128, QUINT8, QINT8, QINT32, STRING, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

If the computation workload is too large, the operator execution may time out (an aicore error is reported, with errorStr: timeout or trap error). The scenario is that the combined size of the last 2 axes is less than 16, while the combined size of the preceding axes is extremely large.

## Invocation Description

| Invocation Method | Invocation Sample                                                   | Description                                               |
|--------------|--------------------------------------------------------|--------------------------------------------------|
| aclnn API | [test_aclnn_eq_tensor.cpp](../equal/examples/test_aclnn_eq_tensor.cpp) | Invokes the Equal operator through the [aclnnEqTensor](../equal/docs/aclnnEqTensor&aclnnInplaceEqTensor_en.md) API. |
| GE graph mode | [test_geir_equal.cpp](examples/test_geir_equal.cpp) | Invokes the Equal operator by building a graph through the [operator IR](op_graph/equal_proto.h).   |
