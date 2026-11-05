# TileWithAxis

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>                                 |    √     |
| <term>Atlas A2 products</term>                                 |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                        |    √     |
| <term>Atlas inference products</term>                                |    √     |
| <term>Atlas training products</term>                                |    √     |

## Function Description

- Operator function: duplicates the input tensor data along a specified dimension to expand the output tensor.

- Computation example:
  - Input x = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], shape = (2, 3, 2)
  - axis = 1, tiles = 2
  - Output y = [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]], shape = (2, 6, 2)

- Computation formula:
  The shape of output y is the same as that of input x, except that the axis dimension becomes `x.shape[axis] * tiles`.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| x      | Input          | Input tensor. | BFLOAT16, FLOAT16, FLOAT, INT32, INT8, UINT8 | ND |
| y      | Output          | Output tensor, which has the same data type and format as x. | Same as x | ND |
| axis   | Optional attribute       | Dimension along which data is duplicated. The default value is 1. | INT | - |
| tiles  | Attribute          | Number of duplicates, which must be greater than 0. | INT | - |

## Constraint Description

- axis must be within the dimension range of the input tensor (negative indices are supported, for example, axis = -1 indicates the last dimension).
- tiles must be greater than 0.
- The size of the output tensor in the axis dimension must equal the size of the input tensor in the axis dimension multiplied by tiles.
- The input tensor has no more than 8 dimensions.
- Scalar input (rank = 0) is supported: the output shape is [tiles], that is, the input scalar value is duplicated tiles times. In this case, axis does not take effect.
- Any dimension of the input tensor can be 0 (empty tensor): the corresponding dimension of the output is still 0, and the output tensor is empty.

## Invocation Description

| Invocation Method   | Sample Code | Description |
|------------|----------|------|
| GE graph mode | [test_geir_tile_with_axis](./examples/test_geir_tile_with_axis.cpp) | Invokes the TileWithAxis operator by building a graph through the [operator IR](./op_graph/tile_with_axis_proto.h). |
