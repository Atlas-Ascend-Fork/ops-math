# If

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

- Operator function: selects and executes the `then_branch` or `else_branch` subgraph based on the value of the Boolean condition `cond`, passes the input tensors to the selected subgraph, and returns its output tensors.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `cond` | Input | Boolean condition for selecting a subgraph branch. | All types | ND |
| `input` | Input (dynamic) | Input tensors passed to the selected subgraph, which support dynamic inputs. | All types | ND |
| `then_branch` | Subgraph | Subgraph executed when the condition is true. It receives `input` and returns a group of tensors. | - | - |
| `else_branch` | Subgraph | Subgraph executed when the condition is false. It receives `input` and returns a group of tensors. | - | - |
| `output` | Output (dynamic) | Output tensors returned by the selected subgraph, which support dynamic outputs. | All types | ND |

## Constraint Description

- The output shape and data type are determined by the subgraph actually executed at runtime, and are not inferred at compilation.
- The output types of the subgraphs in `then_branch` and `else_branch` must be consistent.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_if](./examples/test_geir_if.cpp) | Invokes the If operator by building a graph through the [operator IR](./op_graph/if_proto.h). |
