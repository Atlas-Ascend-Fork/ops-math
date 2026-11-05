# CaseCondition

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: outputs discrete case numbers based on the relationship between three input values and thresholds.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | A 1D tensor with a length of 3, sequentially representing `i`, `j`, and `k`. | INT32, INT64, UINT64 | ND |
| `algorithm` | Attribute | Currently, only `LU` is supported. | STRING | - |
| `y` | Output | Output case number. | INT32 | ND |

## Constraint Description

- `x` must be a vector with shape `[3]`.
- `algorithm` supports only `LU`.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_case_condition](./examples/test_geir_case_condition.cpp) | Invokes the CaseCondition operator by building a graph through the [operator IR](./op_graph/case_condition_proto.h). |
