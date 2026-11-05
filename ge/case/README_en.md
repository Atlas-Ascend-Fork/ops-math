# Case

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

- Operator function: selects a subgraph branch for execution based on the value of `branch_index`, passes the input tensors to the selected subgraph, and returns its output tensors.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `branch_index` | Input | Index for selecting the subgraph branch, which is an INT32 scalar. | INT32 | ND |
| `input` | Input (dynamic) | Input tensors passed to the selected subgraph. Dynamic inputs are supported. | All types | ND |
| `branches` | Subgraph (dynamic) | List of subgraph branches. Each subgraph receives `input` and returns a group of tensors. | - | - |
| `output` | Output (dynamic) | Output tensors returned by the selected subgraph. Dynamic outputs are supported. | All types | ND |

## Constraint Description

- `branch_index` must be an INT32 scalar (rank 0).
- The output types of the subgraphs in `branches` must be consistent.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_case](./examples/test_geir_case.cpp) | Invokes the Case operator by building a graph through the [operator IR](./op_graph/case_proto.h). |
