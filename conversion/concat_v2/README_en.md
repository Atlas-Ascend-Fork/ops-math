# ConcatV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|√|
|Atlas inference products|√|
|Atlas training products|√|

## Function Description

- Operator function: concatenates multiple input tensors along the specified dimension, and outputs a tensor containing all input data concatenated in order.
- Computation process:
  - Input:
    - Tensor list x[0], x[1],..., x[N-1]
    - Concatenation dimension concat_dim

  - Process:
    1. Check that all input tensors have the same data type;
    2. Check that all dimensions except concat_dim are identical;
    3. Concatenate along the concat_dim dimension in sequence:
       y = ConcatV2(x[0], x[1], ..., x[N-1], axis = concat_dim)
  - Output: concatenated tensor y

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
      <td>x</td>
      <td>Input</td>
      <td>Dynamic input list, which is the input x[i] in the process diagram.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, UINT8, INT16, INT8, COMPLEX64, INT64, QINT8, QUINT8, QINT32, UINT16, COMPLEX128, UINT32, UINT64, QINT16, QUINT16, BOOL, STRING, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>Input</td>
      <td>Specifies the concatenation dimension, which is concat_dim in the computation process.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Optional attribute</td>
      <td>Number of input x tensors. The default value is 1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the computation process.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, UINT8, INT16, INT8, COMPLEX64, INT64, QINT8, QUINT8, QINT32, UINT16, COMPLEX128, UINT32, UINT64, QINT16, QUINT16, BOOL, STRING, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

- All input tensors must have the same shape except for the concatenation dimension.
- The input list "x" contains at least two tensors.
- The concatenation dimension concat_dim must be within the valid dimension range of the input tensors.
- All tensors in x must have the same data type.
- The attribute N specifies the number of inputs and is only used for description. It does not affect the dynamic input list at runtime.

## Invocation Description

| Invocation Method | Invocation Sample                                                    | Description                                                           |
|--------------|---------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_concat_v2](./examples/test_geir_concat_v2.cpp) | Invokes the ConcatV2 operator by building a graph through the [operator IR](./op_graph/concat_v2_proto.h). |
