# Eye

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

Operator function: generates a tensor with element values of 1 on the diagonal and 0 elsewhere. If batch_shape is not specified, the output is two-dimensional (num_rows, num_columns); if batch_shape is specified, the output is a multi-dimensional tensor of batch_shape + (num_rows, num_columns), where each two-dimensional slice has 1 on the diagonal and 0 elsewhere.

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
      <td>num_rows</td>
      <td>Attribute</td>
      <td>Indicates the first dimension of the output tensor, an integer on the host. The value range is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_columns</td>
      <td>Optional Attribute</td>
      <td>Indicates the second dimension of the output tensor, an integer on the host. A value not greater than 0 indicates the same as num_rows. The default value is 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_shape</td>
      <td>Optional Attribute</td>
      <td>Indicates the batch dimensions of the output tensor, an integer list on the host. Each item must be greater than 0, and the default value is [] (the output is two-dimensional).</td>
      <td>INT64 (ListInt)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Optional Attribute</td>
      <td>Indicates the data type of the output tensor, an integer on the host, corresponding to a data type enumeration value. The default value is 0 (FLOAT32). The supported data types are the same as those of y.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Indicates the output tensor, with element values of 1 on the diagonal and 0 elsewhere. The shape is batch_shape + (num_rows, num_columns), or two-dimensional (num_rows, num_columns) if batch_shape is not specified.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, UINT8, INT8, INT16, INT32, INT64, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_eye](./examples/test_aclnn_eye.cpp) | Invokes the Eye operator through the [aclnnEye](./docs/aclnnEye_en.md) API. For the API parameters (n, m, out) and constraints, see the API document. |
| Graph mode invocation | - | For the graph IR definition, see the [Eye prototype](./op_graph/eye_proto.h). The operator prototype is `Eye(num_rows, num_columns=0, batch_shape=[], dtype=0) -> y`. |
