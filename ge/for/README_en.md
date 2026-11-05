# For

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

- Operator function: repeatedly executes the `body` subgraph. The loop variable starts from `start` and is incremented by `delta` in each iteration until it exceeds `limit`. In each iteration, `input` is passed to the `body` subgraph, the output returned by `body` serves as the input of the next iteration, and the final output is the result of the last iteration.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `start` | Input | Lower bound of the loop, which is an int32 scalar. | INT32 | ND |
| `limit` | Input | Upper bound of the loop, which is an int32 scalar. | INT32 | ND |
| `delta` | Input | Step size, which is an int32 scalar. | INT32 | ND |
| `input` | Input (dynamic) | Input tensor, which supports dynamic inputs. | All types | ND |
| `body` | Subgraph | Subgraph that receives `input` and returns another group of tensors, used to compute the value of the next iteration. | - | - |
| `output` | Output (dynamic) | Output tensor returned by the `body` subgraph. Its type is the same as that of `input`, and dynamic outputs are supported. | All types | ND |

## Constraint Description

- `start`, `limit`, and `delta` must be INT32 scalars.
- The number of `input` tensors must be the same as the number of `output` tensors.
- The data type of each `input` must be consistent with that of the corresponding `output`.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_for](./examples/test_geir_for.cpp) | Invokes the For operator by building a graph through the [operator IR](./op_graph/for_proto.h). |
