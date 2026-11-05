# Pad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: pads the beginning and end of each dimension of input x with a constant according to the configuration of the input paddings, and the padding value is 0. This operator is compatible with the Pad operator of TensorFlow.
- Example:

  ```text
  Input x = [[0,1,2]], shape is [1,3]
  paddings = [[1,1],[2,2]]

  Output y = [[0,0,0,0,0,0,0],
            [0,0,0,1,2,0,0],
            [0,0,0,0,0,0,0]], shape is [3,7]
  ```

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
      <td>Original tensor to be padded.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, DOUBLE, INT32, UINT8, INT16, INT8, COMPLEX64, INT64, QINT8, QUINT8, QINT32, QINT16, QUINT16, UINT16, COMPLEX128, UINT32, UINT64, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Configuration of padding. The shape is [rank(x), 2], and row i [a, b] indicates that a elements are padded at the beginning of dimension i of x and b elements are padded at the end.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after padding. Its data type is the same as that of x.</td>
      <td>Same as the data type of x</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Data types supported by the AICore path vary with the platform:
  - <term>Atlas A2 products</term>: FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, UINT16, INT64;
  - Some new-generation platforms additionally support the following types: UINT8, UINT32, UINT64, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2;
  - Other platforms: FLOAT, FLOAT16, BFLOAT16, INT16, INT32, UINT16, INT64.
- Data types not in the preceding AICore support list (such as DOUBLE, COMPLEX64, COMPLEX128, and QINT8) are computed through the AICPU path.

## Constraint Description

- The shape of paddings must be [rank(x), 2], where rank(x) is the number of dimensions of input x.
- If the data type of x is FLOAT4_E2M1 or FLOAT4_E1M2, the value of paddings should be an even number.
- If the data type of x is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT8_E8M0, the value of paddings should be a non-negative integer.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|----|----|----|
| GE graph mode | [test_geir_pad](./examples/test_geir_pad.cpp) | Invokes the Pad operator by building a graph through the [operator IR](./op_graph/pad_proto.h). |
