# SwitchByIndex

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: writes the input data at the specified index to the specified register, for label switching at runtime. SwitchByIndex receives the int32 index tensor `x1` and the one-dimensional uint64 tensor `x2`, and has no output tensor.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x1 | Input | Index tensor, whose shape is (1,). | int32 | ND |
| x2 | Input | A one-dimensional tensor. | uint64 | ND |

## Constraint Description

- The shape of `x1` must be (1,), and the data type must be int32.
- `x2` must be a one-dimensional tensor, and its data type must be uint64.
- This operator has no output tensor, and is used only as an internal switching node in a graph. Standalone execution is not yet supported, and the invocation sample only verifies graph building and shape inference.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_switch_by_index](examples/test_geir_switch_by_index.cpp) | Invokes the SwitchByIndex operator by building a graph through the [operator IR](op_graph/switch_by_index_proto.h). |
