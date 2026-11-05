# FlattenV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>                                 |    √     |
| <term>Atlas A2 products</term>                                 |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                        |    √     |
| <term>Atlas inference products</term>                                |    √     |
| <term>Atlas training products</term>                                |    √     |

## Function Description

- Operator function: flattens the dimensions within the `[axis, end_axis]` range of the input tensor into one dimension. The dimensions before `axis` and after `end_axis` are preserved in the output, and the underlying data is not changed. The operator is executed by the graph engine on the host side (Host CPU) and is processed as a memory reuse (reuse-input) operator. The output directly views the input memory.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| axis | Optional attribute | Start axis for flattening. The dimensions before it are preserved in the output. The default value is 1. | INT | - |
| end_axis | Optional attribute | End axis for flattening. The dimensions after it are preserved in the output. The default value is -1 (the last dimension). | INT | - |
| y | Output | Output tensor after flattening. Its data type is the same as `x`. | Same as `x` | ND |

## Constraint Description

- `axis` and `end_axis` must be within the dimension range of the input, and `axis <= end_axis`.
- If there is an unknown dimension (-1) within the `[axis, end_axis]` range, the flattened dimension is -1.
- This operator can be used only as a node in a graph.
- For supported data types, refer to the IR definitions in [`flatten_v2_proto.h`](op_graph/flatten_v2_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_flatten_v2](examples/test_geir_flatten_v2.cpp) | Invokes the FlattenV2 operator by building a graph through the [operator IR](op_graph/flatten_v2_proto.h). |
