# CircularPad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: pads the input tensor in a circular padding mode. This operator is a single GEIR operator, and the 2d/3d mode is determined by the length of paddings.
- CircularPad2d: performs circular padding on the last two dimensions of the input tensor.
- CircularPad3d: performs circular padding on the last three dimensions of the input tensor.

## Parameter Description

<table style="table-layout: fixed; width: 974px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 251px">
<col style="width: 96px">
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
    <td>x</td>
    <td>Input</td>
    <td>Original input data to be padded. The shape supports 3 to 5 dimensions: in 2d mode, the last 2 dimensions are padded; in 3d mode, the last 3 dimensions are padded.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>paddings</td>
    <td>Input</td>
    <td>Padding configuration. When the length is 4, the values sequentially represent the padding values for left, right, top, and bottom (2d mode); when the length is 6, the values sequentially represent the padding values for left, right, top, bottom, front, and back (3d mode).</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output result after padding, with the same number of dimensions as x.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraint Description

- For CircularPad of Ascend 950 (Arch35), the positive padding can be equal to the corresponding size used for verification (the valid size after negative padding cropping in graph mode, and the input dimension for the ACLNN API); wrapping around for multiple loops beyond this size is not supported. For other products, refer to the constraints in the corresponding version documentation.
- The size of the last dimension of y under different types must meet the following constraints (applicable only to Atlas A2/A3 and Kirin series products; Ascend 950/960/350 (Arch35) has no such upper limit constraint):
  - int8: (0, 98304)
  - float16/bfloat16: (0, 49152)
  - int32/float32: (0, 24576)
- The data types of the input and output must be the same.
- In 2d mode (paddings length 4), x must have 3 or 4 dimensions; in 3d mode (paddings length 6), x must have 4 or 5 dimensions. That is, the length of paddings must equal 2 × the number of dimensions to be padded (the last 2 dimensions in 2d mode, and the last 3 dimensions in 3d mode).

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_circular_pad](./examples/test_aclnn_circular_pad.cpp) | Invokes the CircularPad operator through the [aclnnCircularPad2d](docs/aclnnCircularPad2d_en.md) and [aclnnCircularPad3d](docs/aclnnCircularPad3d_en.md) APIs. |
| Graph-mode invocation | [test_geir_circular_pad](./examples/test_geir_circular_pad.cpp) | Invokes the CircularPad operator by building a graph through the [operator IR](./op_graph/circular_pad_proto.h). |
