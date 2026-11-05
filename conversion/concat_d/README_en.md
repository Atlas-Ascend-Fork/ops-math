# ConcatD

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                       |    √     |
| Atlas A3 products     |    √     |
| Atlas A2 products      |    √     |
| Atlas 200I/500 A2 inference products                      |    √     |
| Atlas inference products                              |    √     |
| Atlas training products                              |    √     |
| Kirin X90 processor series products                       |    √     |
| Kirin 9030 processor series products                       |    √     |

## Function Description

- Operator function: concatenates all tensors in tensors along dimension dim. All dimensions except the dimension corresponding to dim must be the same.

## Parameter Description

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>List of tensors to be concatenated.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, BFLOAT16, DOUBLE, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concat_dim</td>
      <td>Attribute</td>
      <td>Dimension along which tensors are concatenated.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Optional attribute</td>
      <td>Number of tensors to be concatenated.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, BFLOAT16, DOUBLE, COMPLEX64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E1M2, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products, Atlas inference products, and Atlas 200I/500 A2 inference products: BFLOAT16 is not supported.
- Atlas A2 products and Atlas A3 products: UINT16, UINT32, UINT64, FLOAT4_E1M2, and FLOAT4_E2M1 are not supported.
- Kirin X90/9030 processor series products: The data types FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, and UINT64 are supported; 1 to 32 non-empty ND inputs are supported, the input rank is known at compile time and is 1 to 8, and each dimension must be resolved to a positive number before tiling at runtime.

## Constraint Description

- The data types and data formats of elements in the x list must be within the supported range.
- Data type inference can be performed on the x list.
- The inferred data type must be convertible to the specified output type of y.
- The shapes of non-concatenated dimensions must be the same.
- dim must be within the dimension range of x.
- When the input data type is FLOAT4_E1M2 or FLOAT4_E2M1, the product of the sizes of each input tensor along the concat axis and the dimensions after it must be an even number.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_cat](examples/test_aclnn_cat.cpp) | Invokes the ConcatD operator through the [aclnnCat](docs/aclnnCat_en.md) API. |
