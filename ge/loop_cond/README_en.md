# LoopCond

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

- Operator function: represents the termination condition of a while loop. It forwards the Boolean scalar input tensor `x` as-is to the output tensor `y` to pass the loop termination condition to the Switch operator. It works with operators such as Enter, Exit, Switch, and NextIteration to form a loop control structure.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Boolean scalar, which is the loop termination condition. | bool | ND |
| y | Output | Tensor that has the same shape, data type, and content as the input `x`. | Same as `x` | ND |

## Constraint Description

- The input `x` must be a scalar (rank 0, or a dimension of -2 in the unknown-rank scenario).
- The data type of the input `x` must be bool.
- Dynamic shape inputs containing -1 are not supported (-1 indicates rank 1, which does not satisfy the scalar constraint, and graph building verification fails).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_loop_cond](examples/test_geir_loop_cond.cpp) | Invokes the LoopCond operator by building a graph through the [operator IR](op_graph/loop_cond_proto.h). |
