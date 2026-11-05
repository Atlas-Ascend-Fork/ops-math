# Size

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

- Operator function: returns the size of the input tensor, that is, an integer scalar of the number of elements in the tensor.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | Input tensor. | All types | ND |
| `dtype` | Attribute | Optional integer, which is the output data type. The value is `DT_INT32` or `DT_INT64`, and the default value is `DT_INT32`. | Int | - |
| `y` | Output | Number of elements in the input tensor, which is a scalar. | INT32, INT64 | ND |

## Constraint Description

- The value of the `dtype` attribute supports only `DT_INT32` and `DT_INT64`.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_size](./examples/test_geir_size.cpp) | Invokes the Size operator by building a graph through the [operator IR](./op_graph/size_proto.h). |
