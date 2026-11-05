# StopGradient

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

- Operator function: passes the input tensor to the output as is, and prevents the backpropagation computation from passing through this node.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| y | Output | Tensor that has the same shape, data type, and content as the input `x`. | Same as `x` | ND |

## Constraint Description

- The shape and data type of `x` and `y` are the same.
- This operator does not perform numerical transformation, and only changes the gradient propagation behavior.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_stop_gradient](examples/test_geir_stop_gradient.cpp) | Invokes the StopGradient operator by building a graph through the [operator IR](op_graph/stop_gradient_proto.h). |
