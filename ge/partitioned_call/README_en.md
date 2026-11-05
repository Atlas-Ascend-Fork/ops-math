# PartitionedCall

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

- Operator function: passes the input tensors `args` to the subgraph `f` and returns the output tensors of the subgraph `f`. The subgraph `f` receives the inputs and returns another group of tensors.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `args` | Input (dynamic) | Input tensors, which will be passed to the subgraph `f` and support dynamic inputs. | All types | ND |
| `f` | Subgraph | Subgraph that receives `args` and returns another group of tensors. | - | - |
| `config` | Attribute | Optional string, which is empty by default. | String | - |
| `config_proto` | Attribute | Optional string, which is empty by default. | String | - |
| `executor_type` | Attribute | Optional string, which is empty by default. | String | - |
| `output` | Output (dynamic) | Output tensors returned by the subgraph `f`, which support dynamic outputs. | All types | ND |

## Constraint Description

- The number of `args` tensors must be the same as the number of `output` tensors.
- The data type of each `args` tensor must be consistent with that of the corresponding `output`.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_partitioned_call](./examples/test_geir_partitioned_call.cpp) | Invokes the PartitionedCall operator by building a graph through the [operator IR](./op_graph/partitioned_call_proto.h). |
