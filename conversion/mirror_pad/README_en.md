# MirrorPad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                        |    √     |
| <term>Atlas A3 products</term>        |    √     |
| <term>Atlas A2 products</term>        |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                        |    ×     |
| <term>Atlas inference products</term>                                |    √     |
| <term>Atlas training products</term>                                |    √     |

## Function Description

- Operator function: pads the boundaries of the input tensor in a mirrored manner.
- Example:

  ```text
  Input tensor([[0,1,2]])
  paddings([2,2])

  1.mode("REFLECT")
  The output is ([[2,1,0,1,2,1,0]])

  2.mode("SYMMETRIC")
  The output is ([[1,0,0,1,2,2,1]])
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
      <td>Original tensor to be mirrored and expanded.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Configuration of the expansion.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attribute</td>
      <td>Padding mode, which can be reflection "REFLECT" or symmetry "SYMMETRIC".</td>
      <td>string</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after expansion.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BF16, FLOAT16, FLOAT, DOUBLE, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

Note: HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT8_E8M0 are supported only on <term>Ascend 950PR&950DT products</term> and are not supported on other products.

- paddings
  Dimension 0 of paddings indicates the expansion configuration for dimension 0 of input x, and so on. Each row indicates the number of paddings in the corresponding dimension (left/right, front/back, top/bottom, and so on). For each row [a, b], a indicates the number of elements padded at the beginning of the dimension, and b indicates the number of elements padded at the end of the dimension.

- mode
  In REFLECT mode, the boundary itself is not included during mirroring.
  In SYMMETRIC mode, the boundary itself is included during mirroring, as shown in the example.

## Constraint Description

The shape of paddings must be [rank, 2], where rank is the number of dimensions of input x.
For each row paddings[i], the number of padded elements cannot exceed the number of elements of the corresponding x[i]. That is, in reflection mode, paddings[i][0] and paddings[i][1] <= x[i].length-1; in symmetric mode, paddings[i][0] and paddings[i][1] <= x[i].length.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|----|----|----|
| aclnn invocation | [test_aclnn_reflection_pad_1d](./examples/test_aclnn_reflection_pad_1d.cpp) | Invokes the mirror_pad operator in the REFLECT case through the [aclnnReflectionPad1d](./docs/aclnnReflectionPad1d_en.md) API, and pads the last dimension of the input tensor. |
| aclnn invocation | [test_aclnn_reflection_pad_2d](./examples/test_aclnn_reflection_pad_2d.cpp) | Invokes the mirror_pad operator in the REFLECT case through the [aclnnReflectionPad2d](./docs/aclnnReflectionPad2d_en.md) API, and pads the last two dimensions of the input tensor. |
| aclnn invocation | [test_aclnn_reflection_pad_3d](./examples/test_aclnn_reflection_pad_3d.cpp) | Invokes the mirror_pad operator in the REFLECT case through the [aclnnReflectionPad3d](./docs/aclnnReflectionPad3d_en.md) API, and pads the last three dimensions of the input tensor. |
