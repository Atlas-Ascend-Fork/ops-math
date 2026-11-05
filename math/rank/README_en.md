# Rank

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

- Operator function: returns the rank of the input tensor, that is, the number of dimensions of the tensor. The output is an INT32 scalar.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | Input tensor. | All types | ND |
| `y` | Output | Number of dimensions of the input tensor, a scalar. | INT32 | ND |

## Constraint Description

- The input `x` supports all data types and any shape (including scalars and dynamic shapes).
- The output `y` is always a 0-dimension scalar, the data type is fixed to INT32, and it does not change with the input (including inputs with unknown rank -2).
- Inputs support only the ND format.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_rank](./examples/test_geir_rank.cpp) | Invokes the Rank operator by building a graph through the [operator IR](./op_graph/rank_proto.h). |
