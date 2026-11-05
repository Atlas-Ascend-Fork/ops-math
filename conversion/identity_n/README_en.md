# IdentityN

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

- Operator function: returns copies of a group of input tensors. The outputs have the same number, shape, data type, and content as the inputs.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | List of input tensors to be copied. Dynamic input. | FLOAT, FLOAT16, INT8, INT16, UINT16, UINT8, INT32, INT64, UINT32, UINT64, BOOL, DOUBLE, STRING | ND |
| y | Output | Copies of the input tensor list `x`. Dynamic output. The length, shape, data type, and content are consistent with `x`. | Same as `x` | ND |

## Constraint Description

- The number, shape, data type, and element content of `x` and `y` remain consistent.
- For supported data types, refer to the IR definitions in [`identity_n_proto.h`](./op_graph/identity_n_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_identity_n](examples/test_geir_identity_n.cpp), [test_geir_identity_n_v2](examples/test_geir_identity_n_v2.cpp) | Invokes the IdentityN operator by building a graph through the [operator IR](op_graph/identity_n_proto.h). |
