# Cast

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |
| <term>Kirin X90 processor series products</term>                              |    √     |
| <term>Kirin 9030 processor series products	</term>                              |    √     |

## Function Description

Converts the input tensor to the specified dtype.

## Parameter Description

<table style="table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 200px">
  <col style="width: 460px">
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
      <td>Input parameter for the cast computation.</td>
      <td>BOOL, FLOAT16, FLOAT, INT8, INT32, UINT32, UINT8, INT64, UINT64, INT16, UINT16, DOUBLE, COMPLEX64, COMPLEX128, QINT8, QUINT8, QINT16, QUINT16, QINT32, BF16, UINT1, COMPLEX32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for the cast computation.</td>
      <td>BOOL, FLOAT16, FLOAT, INT8, INT32, UINT32, UINT8, INT64, UINT64, INT16, UINT16, DOUBLE, COMPLEX64, COMPLEX128, QINT8, QUINT8, QINT16, QUINT16, QINT32, BF16, UINT1, COMPLEX32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.
- Atlas A2 products and Atlas A3 products: COMPLEX32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.

## Constraint Description

- For the scenario where the data type is converted from a floating-point number to an integer:
  If nan exists in the input data, nan is converted to 0.

- For the scenario where the input data type is BOOL, COMPLEX32, COMPLEX64, COMPLEX128, FLOAT4_E2M1, or FLOAT4_E1M2:
  Non-contiguous inputs are not supported.

- <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>:
  - For the scenario where the data type is converted from INT32 to INT8:
    Accuracy is error-free only when the input data is in the range (-2048, 1920).
  - For the scenario where the data type is converted from FLOAT64/COMPLEX64/COMPLEX128 to UINT8:
    Accuracy is error-free only when the input data is a non-negative number.

- <term>Atlas inference products</term>:
  - For the scenario where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8:
    Accuracy is error-free only when the input data is in the range (-2147483648, 2147483583).

  - For the scenario where the data type is converted from INT64 to FLOAT32:
    Accuracy is error-free only when the input data is in the range (-2147483648, 2147483647).

- <term>Ascend 950PR&950DT products</term>:
  - For input and output types involving COMPLEX32, COMPLEX64, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT8_E8M0, only the conversion paths in the following table are supported:

    | `self` data type | `out` data type |
    | -------- | -------- |
    | COMPLEX32 | COMPLEX64 |
    | COMPLEX64 | COMPLEX32 |
    | COMPLEX32 | FLOAT16 |
    | FLOAT16 | COMPLEX32 |
    | FLOAT32/BFLOAT16 | COMPLEX64 |
    | COMPLEX64 | FLOAT32/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | FLOAT32/FLOAT16/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32/FLOAT16/BFLOAT16 |
    | BFLOAT16 | FLOAT8_E8M0 |
    | FLOAT8_E8M0 | BFLOAT16 |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_cast](./examples/test_aclnn_cast.cpp) | Invokes the Cast operator through the [aclnnCast](./docs/aclnnCast_en.md) API. |
| Graph-mode invocation | [test_geir_cast](./examples/test_geir_cast.cpp)   | Invokes the Cast operator by building a graph through the [operator IR](./op_graph/cast_proto.h). |
