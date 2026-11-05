# TransShape

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

- Operator function: changes the shape of the output based on the `outShape` attribute without changing the underlying data. During graph preparation, GE replaces the `TransShape` node with a `MemcpyAsync` node to perform the data copy.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| outShape | Optional attribute | Shape of the output. The default value is an empty list (scalar). | LIST_INT | - |
| y | Output | Output tensor, whose shape equals `outShape` and whose data type is the same as that of `x`. | Same as `x` | ND |

## Constraint Description

- The product of the dimension values of `outShape` must be consistent with the total number of elements in the input `x`. Otherwise, the runtime data copy fails due to a memory mismatch.
- The supported data types are subject to the IR definitions in [`trans_shape_proto.h`](op_graph/trans_shape_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_trans_shape](examples/test_geir_trans_shape.cpp) | Invokes the TransShape operator by building a graph through the [operator IR](op_graph/trans_shape_proto.h). |
