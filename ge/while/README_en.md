# While

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

- Operator function: executes the `body` subgraph repeatedly until the `cond` subgraph returns false. Before each iteration, `cond` is executed first to determine whether to continue. If the loop continues, the tensors returned by the `body` subgraph are used as the loop variables of the next iteration. When the loop ends, the current loop variables are output.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `input` | Input (dynamic) | Input tensors of loop variables, which are passed between iterations through the output of the `body` subgraph. | All types | ND |
| `cond` | Subgraph | The loop condition subgraph, which receives the loop variables and returns a Boolean scalar. The loop ends when the value is false. | - | - |
| `body` | Subgraph | The loop body subgraph, which receives the loop variables and returns a group of tensors as the loop variables of the next iteration. | - | - |
| `parallel_iterations` | Attribute | The number of parallel iterations, which controls the execution mode of iterations at runtime. The default value is 10. | Int | - |
| `output` | Output (dynamic) | Output tensors of loop variables returned by the `body` subgraph after the loop ends. | All types | ND |

## Constraint Description

- The output shape and data type are determined by the loop iterations actually executed at runtime, and are not inferred at compile time.
- The output of the `cond` subgraph must be a Boolean scalar. The number, shapes, and data types of the outputs of the `body` subgraph must be the same as those of the input loop variables.
- If `cond` returns false on the first execution, the loop body is not executed, and the output is the same as the input tensors.

## Invocation Description

The While operator is a basic control flow operator, and is used by the framework in loop control scenarios. It is not invoked separately. For details about the operator interface, see the [operator IR](./op_graph/while_proto.h) definition.
