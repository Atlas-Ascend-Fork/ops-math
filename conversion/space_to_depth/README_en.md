# SpaceToDepth

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                             | √        |
| <term>Atlas A3 products</term>     | √        |
| <term>Atlas A2 products</term> | √        |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×     |
| <term>Atlas inference products</term>                             | √        |
| <term>Atlas training products</term>                              | √        |

## Function Description

- Operator function: moves data in the spatial dimensions (height and width) to the depth (channel) dimension.

- Function Description:
  This operator rearranges the spatial dimensions (height and width) of the input tensor and converts them into the depth (channel) dimension. Specifically, it divides the height and width dimensions of the input tensor into blocks of the specified block size (block_size), and rearranges these spatial blocks into the channel dimension. As a result, the size of the spatial dimensions decreases, while the depth of the channel dimension increases.

- Computation formula:
  Assume that the input tensor shape is:
  - If data_format="NHWC": `[batch, height, width, channels]`
  - If data_format="NCHW": `[batch, channels, height, width]`

  Assume that block_size is B. The output tensor shape is:
  - If data_format="NHWC": `[batch, height/B, width/B, channels*B*B]`
  - If data_format="NCHW": `[batch, channels*B*B, height/B, width/B]`

  B must integer-divide both the height and width dimensions.

## Parameter Description

<table style="table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, which supports multiple data types. The data format must be the same as the format specified by the data_format attribute</td>
      <td>FLOAT, FLOAT16, INT8, INT16, INT32, UINT8, UINT16, UINT32, INT64, UINT64, BFLOAT16</td>
      <td>NHWC, NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which has the same data type as the input x. The output shape is computed based on block_size and data_format</td>
      <td>Same as x</td>
      <td>NHWC, NCHW</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute (Required)</td>
      <td>Size of the spatial block. This parameter must be an integer greater than 1. The height and width dimensions must be divisible by block_size</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute (Optional)</td>
      <td>Data format, which specifies the dimension order of the input and output. Both "NHWC" and "NCHW" formats are supported, and the default value is "NHWC"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  Graph-mode invocation  |[test_geir_space_to_depth](examples/test_geir_space_to_depth.cpp) | Invokes the SpaceToDepth operator by building a graph through the [operator IR](./op_graph/space_to_depth_proto.h). |
