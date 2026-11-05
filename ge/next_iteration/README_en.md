# NextIteration

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

- Operator function: makes the input available in the next iteration. It passes the input tensor `x` as-is to the output tensor `y` to pass the computation result of the current iteration to the next iteration in a while loop. It works with operators such as Enter, Exit, Switch, and LoopCond to form a loop control structure.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Tensor that needs to be available in the next iteration. | float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool | ND |
| y | Output | Tensor that has the same shape, data type, and content as the input `x`. | Same as `x` | ND |

## Constraint Description

- The shapes and data types of `x` and `y` are consistent.
- This operator is a while loop control flow operator and must be used within a complete loop structure formed by Enter, Merge, Switch, and LoopCond. It cannot be executed when a graph is built separately outside the loop structure. The invocation sample only verifies graph building and shape inference.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_next_iteration](examples/test_geir_next_iteration.cpp) | Invokes the NextIteration operator by building a graph through the [operator IR](op_graph/next_iteration_proto.h). |
