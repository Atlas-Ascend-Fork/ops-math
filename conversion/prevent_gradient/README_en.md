# PreventGradient

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

- Operator function: during forward computation, returns the input tensor as is without any numerical transformation; during backpropagation (gradient computation), this operator does not provide a gradient. Once gradient computation needs to flow through this operator, the framework immediately reports an error and terminates. This operator is used to explicitly block gradient backpropagation.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| y | Output | Tensor with the same shape, data type, and content as the input `x`. | Same as `x` | ND |
| message | Attribute | Prompt message output when an error is reported during backpropagation. The default value is an empty string. | STRING | - |

## Constraint Description

- The shapes and data types of `x` and `y` are consistent.
- This operator does not perform numerical transformation during forward computation; `message` is output only as a prompt message when an error is reported during backpropagation.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_prevent_gradient](examples/test_geir_prevent_gradient.cpp) | Invokes the PreventGradient operator by building a graph through the [operator IR](op_graph/prevent_gradient_proto.h). |
