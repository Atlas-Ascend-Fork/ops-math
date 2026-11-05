# DepthToSpace

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :---------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: moves the data of the depth (channel) dimension to the spatial dimensions (height and width). This operator rearranges the depth (channel) dimension of the input tensor and converts it into the spatial dimensions (height and width). Specifically, it divides the depth dimension of the input tensor according to the specified block size (block_size) and rearranges these depth blocks into the spatial dimensions. As a result, the size of the spatial dimensions increases and the depth of the channel dimension decreases.

- Computation formula:
  Suppose the shape of the input tensor is:
  - When data_format="NHWC": `[batch, height, width, channels]`
  - When data_format="NCHW": `[batch, channels, height, width]`

  Suppose block_size is B. The shape of the output tensor is:
  - When data_format="NHWC": `[batch, height*B, width*B, channels/(B*B)]`
  - When data_format="NCHW": `[batch, channels/(B*B), height*B, width*B]`

  Here, the channel dimension must be divisible by B*B.

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
      <td>Input tensor that supports multiple data types. The data format must be consistent with the format specified by the data_format attribute</td>
      <td>FLOAT, FLOAT16, INT8, INT16, INT32, UINT8, UINT16, UINT32, INT64, UINT64, BFLOAT16</td>
      <td>NHWC, NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, which has the same data type as the input x. The output shape is computed based on block_size, mode, and data_format</td>
      <td>Same as x</td>
      <td>NHWC, NCHW</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute (Required)</td>
      <td>Size of the spatial block. It must be an integer greater than or equal to 2. The channel dimension must be divisible by the square of block_size</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Attribute (Required)</td>
      <td>Mode of data rearrangement. Both the "DCR" and "CRD" modes are supported</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute (Optional)</td>
      <td>Data format, which specifies the dimension order of the input and output. Both the "NHWC" and "NCHW" formats are supported. The default value is "NHWC"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraint Description

- The dimensions of the input and output tensors must be 4D
- The input and output must have the same data format, and only the NCHW and NHWC formats are supported
- block_size must be an integer greater than or equal to 2
- The channel dimension must be divisible by the square of block_size
- The mode attribute must be "DCR" or "CRD"
- The data_format attribute must be consistent with the actual format of the input tensor

## Invocation Description

| Invocation Method  | Sample Code   | Description    |
| :-------- | :---------- | :------------------------ |
|  Graph-mode invocation  |[test_geir_depth_to_space](examples/test_geir_depth_to_space.cpp) | Invokes the DepthToSpace operator by building a graph through the [operator IR](./op_graph/depth_to_space_proto.h). |
