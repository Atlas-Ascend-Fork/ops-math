# Identity

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: returns a tensor `y` that has the same shape and content as the input tensor `x`.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| y | Output | Tensor that has the same shape and content as the input tensor `x`. | Same as `x` | ND |

## Constraint Description

- The shape, data type, and element content of `x` and `y` remain consistent.
- For supported data types, refer to the IR definitions in [`identity_proto.h`](./op_graph/identity_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_identity](examples/test_geir_identity.cpp), [test_geir_identity_v2](examples/test_geir_identity_v2.cpp) | Invokes the Identity operator by building a graph through the [operator IR](op_graph/identity_proto.h). |
