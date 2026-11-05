# Fill

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- **Operator function**: performs a fill operation on tensors and supports non-contiguous Tensor operations.
- **InplaceFillScalar**: fills the entire tensor with a scalar value.
- **InplaceFillTensor**: fills the entire tensor with a tensor value (the value tensor must be a 0-dimensional tensor or a 1-dimensional tensor with size=1).

## Parameter Description

### InplaceFillScalar

<table style="table-layout: fixed; width: 922px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 308px">
<col style="width: 102px">
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
    <td>selfRef</td>
    <td>Input/Output tensor</td>
    <td>Input and output tensor. Non-contiguous Tensors are supported, and data dimensions above 8 are not supported.</td>
    <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, COMPLEX64, COMPLEX128, BOOL, BFLOAT16, UINT16, UINT32, UINT64, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input scalar</td>
    <td>Specified scalar value. Its data type must satisfy the data type inference rules with the data type of selfRef.</td>
    <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, COMPLEX64, COMPLEX128, BOOL, BFLOAT16, UINT16, UINT32, UINT64, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, HIFLOAT8</td>
    <td>-</td>
  </tr>
</tbody>
</table>

### InplaceFillTensor

<table style="table-layout: fixed; width: 922px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 202px">
<col style="width: 308px">
<col style="width: 102px">
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
    <td>selfRef</td>
    <td>Input/Output tensor</td>
    <td>Input and output tensor. Non-contiguous Tensors are supported, and data dimensions above 8 are not supported.</td>
    <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, COMPLEX64, COMPLEX128, BOOL, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input tensor</td>
    <td>Fill value tensor. It must be a 0-dimensional tensor or a 1-dimensional tensor with size=1.</td>
    <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, COMPLEX64, COMPLEX128, BOOL, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, HIFLOAT8</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- Data dimensions above 8 are not supported for the input tensor.
- The value parameter of InplaceFillTensor must be a 0-dimensional tensor or a 1-dimensional tensor with size=1.
- Data types must satisfy the mutual inference relationship.
- The FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, and HIFLOAT8 data types are supported only on <term>Ascend 950PR&950DT products</term> and are not supported on other products.
- For CANN 9.0.0, the FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT8_E8M0, and HIFLOAT8 data types are not supported.

## Invocation Description

| Invocation Method  | Sample Code                                              | Description                                                         |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_fill_scalar.cpp](./examples/test_aclnn_fill_scalar.cpp) | Invokes the InplaceFillScalar operator through the [aclnnInplaceFillScalar](docs/aclnnInplaceFillScalar_en.md) API. |
| aclnn API | [test_aclnn_fill_tensor.cpp](./examples/test_aclnn_fill_tensor.cpp) | Invokes the InplaceFillTensor operator through the [aclnnInplaceFillTensor](docs/aclnnInplaceFillTensor_en.md) API. |
