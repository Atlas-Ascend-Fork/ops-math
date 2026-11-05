# ControlTrigger

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT products</term>       |   √  |
| <term>Atlas A3 products</term> |   √  |
| <term>Atlas A2 products</term> |   √  |
| <term>Atlas 200I/500 A2 inference products</term>          |   √  |
| <term>Atlas inference products</term>                    |   √  |
| <term>Atlas training products</term>                    |   √  |

## Function Description

- Operator function: serves only as a placeholder for control edges. It is similar to NoOp, but always produces active control outputs even if some control inputs have become invalid.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| ---- | ---- | ---- | ---- | ---- |
| None | - | ControlTrigger has no inputs, outputs, or attributes. | - | - |

## Constraint Description

- ControlTrigger has no data inputs or outputs and supports use only as a placeholder for control edges.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_control_trigger](./examples/test_geir_control_trigger.cpp) | Invokes the ControlTrigger operator by building a graph through the [operator IR](./op_graph/control_trigger_proto.h). |
