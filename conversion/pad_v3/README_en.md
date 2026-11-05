# PadV3

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>       |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function Description

- Operator function: pads the input tensor.
- Example:

  ```text
  Input tensor([[0,1,2]])
  paddings([2,2])

  1.mode("REFLECT")
  The output is ([[2,1,0,1,2,1,0]])

  2.mode("SYMMETRIC")
  The output is ([[1,0,0,1,2,2,1]])

  3.mode("constant")
  The output is ([[0,0,0,1,2,0,0]])

  4.mode("edge")
  The output is ([[0,0,0,1,2,2,2]])
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
      <td>Original tensor to be expanded.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
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
      <td>constant_values</td>
      <td>Input</td>
      <td>Value of the padding constant in constant mode.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attribute</td>
      <td>Padding mode. There are four options: constant "constant", edge "edge", reflection "REFLECT", and symmetry "SYMMETRIC".</td>
      <td>string</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings_contiguous</td>
      <td>Attribute</td>
      <td>Controls whether the storage of the paddings tensor is "contiguous".</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after expansion.</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>: the data types BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the data types HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.

- paddings
    Dimension 0 of paddings indicates the expansion configuration for dimension 0 of input x, and so on. Each row indicates the number of paddings in the corresponding dimension (left/right, front/back, top/bottom, and so on). For each row [a, b], a indicates the number of elements padded at the beginning of the dimension, and b indicates the number of elements padded at the end of the dimension (when paddings_contiguous is true).

- mode
    In constant mode, the tensor is padded with a constant value. The constant value is indicated in the input constant_values, and an empty constant_values indicates padding with 0.
    In edge mode, the tensor is padded with the edge values of input x.
    In REFLECT mode, the tensor is padded in a mirrored manner, and the boundary itself is not included during mirroring.
    In SYMMETRIC mode, the tensor is padded in a mirrored manner, and the boundary itself is included during mirroring.
- paddings_contiguous

    If it is true, paddings is arranged in row-major order.
    For example, [[2,2],[1,1]] indicates:

        Dimension 0: 2 is padded on the left and 2 on the right.
        Dimension 1: 1 is padded on the left and 1 on the right.

    If it is false, paddings is arranged in column-major order.
    For example, [[2,1],[2,1]] indicates:

        Dimension 0: 2 is padded on the left and 2 on the right.
        Dimension 1: 1 is padded on the left and 1 on the right.

## Constraint Description

In the case of paddings_contiguous = true, that is, row-major order:
The shape of paddings must be [rank, 2], where rank is the number of dimensions of input x.
For each row paddings[i], the number of padded elements cannot exceed the number of elements of the corresponding x[i]. That is, in REFLECT mode, paddings[i][0] and paddings[i][1] <= x[i].length-1; in SYMMETRIC mode, paddings[i][0] and paddings[i][1] <= x[i].length.

In the case of paddings_contiguous = false, that is, column-major order, the paddings constraints of REFLECT mode and SYMMETRIC mode are the same by analogy.

In CONSTANT mode, if the data type of x is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, or FLOAT4_E1M2, all dimensions of paddings must be non-negative numbers; if the data type of x is FLOAT4_E2M1 or FLOAT4_E1M2, the paddings of the last dimension of input x must also be even numbers.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|----|----|----|
| aclnn invocation | [test_aclnn_constant_pad_nd](./examples/test_aclnn_constant_pad_nd.cpp) | Invokes the pad_v3 operator in constant mode through the [aclnnConstantPadNd](./docs/aclnnConstantPadNd_en.md) API. |
| aclnn invocation | [test_aclnn_replication_pad_1d](./examples/test_aclnn_replication_pad_1d.cpp) | Invokes the pad_v3 operator in edge mode through the [aclnnReplicationPad1d](./docs/aclnnReplicationPad1d_en.md) API, and pads the last dimension of the input tensor. |
| aclnn invocation | [test_aclnn_replication_pad_2d](./examples/test_aclnn_replication_pad_2d.cpp) | Invokes the pad_v3 operator in edge mode through the [aclnnReplicationPad2d](./docs/aclnnReplicationPad2d_en.md) API, and pads the last two dimensions of the input tensor. |
| aclnn invocation | [test_aclnn_replication_pad_3d](./examples/test_aclnn_replication_pad_3d.cpp) | Invokes the pad_v3 operator in edge mode through the [aclnnReplicationPad3d](./docs/aclnnReplicationPad3d_en.md) API, and pads the last three dimensions of the input tensor. |
