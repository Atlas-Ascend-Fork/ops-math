# Pack

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>      |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: packs a list of tensors in values into a tensor that is one dimension higher than each tensor in values, by packing along the axis dimension. Given a list of length N of tensors with shape (A, B, C); if axis == 0, the output tensor will have shape (N, A, B, C).

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
      <td>COMPLEX128, COMPLEX64, BOOL, DOUBLE, FLOAT32, FLOAT16, BFLOAT16, INT16, INT32, INT64, INT8, UINT8, UINT16, UINT32, UINT64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td>Specifies the dimension along which tensors are packed.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Optional attribute</td>
      <td>Specifies the number of tensors to be packed.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>COMPLEX128, COMPLEX64, BOOL, DOUBLE, FLOAT32, FLOAT16, BFLOAT16, INT16, INT32, INT64, INT8, UINT8, UINT16, UINT32, UINT64, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Atlas training products, Atlas inference products, and Atlas 200I/500 A2 inference products: BFLOAT16, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN are not supported.
* Atlas A2 products and Atlas A3 products: HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN are not supported.

## Constraint Description

None.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_stack](examples/test_aclnn_stack.cpp) | Invokes the Pack operator through the [aclnnStack](docs/aclnnStack_en.md) API. |
