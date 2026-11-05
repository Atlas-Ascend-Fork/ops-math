# Snapshot

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: returns a copy of the input tensor. The output has the same shape, data type, and content as the input.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor to be copied. | FLOAT, FLOAT16, INT8, INT16, UINT16, UINT8, INT32, INT64, UINT32, UINT64, BOOL, DOUBLE, STRING | ND |
| y | Output | Copy of the input tensor `x`. | Same as `x` | ND |

## Constraint Description

- The shape, data type, and element content of `x` and `y` are the same.
- For the supported data types, refer to the IR definition in [`snapshot_proto.h`](./op_graph/snapshot_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_snapshot](examples/test_geir_snapshot.cpp) | Invokes the Snapshot operator by building a graph through the [operator IR](op_graph/snapshot_proto.h). |
