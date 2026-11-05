# RefIdentity

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

- Operator function: The function is the same as [Identity](../../conversion/identity/README_en.md)—the output tensor `y` has the same shape, data type, and content as the input tensor `x`. The difference is that the input of RefIdentity is a reference tensor that can be overwritten by ref-type operators, and is used for reference (ref) type data.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `x` | Input | Input tensor. | float16, float, double, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, bfloat16 | ND |
| `y` | Output | Tensor that has the same shape, data type, and content as the input `x`. | Same as `x` | ND |

## Constraint Description

- The data type and shape of the input `x` and the output `y` are the same.
- Dynamic shape and dynamic rank are supported.

## Invocation Description

The RefIdentity operator has no prototype (REG_OP) or inference implementation yet, and no invocation sample is provided. If you want to contribute a complete implementation of this operator, refer to [Contribution Process](../../CONTRIBUTING_en.md).
