# Eye

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Operator function: returns a two-dimensional tensor with element values of 1 on the diagonal and element values of 0 elsewhere.

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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>n</td>
      <td>Input</td>
      <td>Indicates the first dimension, an integer on the host. The value range is not less than 0.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Input</td>
      <td>Indicates the second dimension, an integer on the host. The value range is not less than 0.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_shape</td>
      <td>Input</td>
      <td>Optional. Indicates the batch dimensions of the output tensor, an integer list on the host. Each item must be greater than 0. If this parameter is not specified, the output is two-dimensional.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Optional. Indicates the data type of the output tensor, an integer on the host, corresponding to a data type enumeration value. The default value is FLOAT32.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Indicates the output tensor, an aclTensor on the device. The shape is batch_shape + (n, m), or two-dimensional (n, m) if batch_shape is not specified.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT32, INT16, INT8, UINT8, INT64, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_eye](./examples/test_aclnn_eye.cpp) | Invokes the Eye operator through the [aclnnEye](./docs/aclnnEye_en.md) API. |
