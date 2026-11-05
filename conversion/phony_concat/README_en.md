# PhonyConcat

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>                                 |    √     |
| <term>Atlas A2 products</term>                                 |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                        |    √     |
| <term>Atlas inference products</term>                                |    √     |
| <term>Atlas training products</term>                                |    √     |

## Function Description

- Operator function: concatenates a group of tensors along the dimension specified by `concat_dim`. All inputs of `PhonyConcat` are allocated on the same contiguous memory, and GE computes the memory offset of each input so that preceding custom operators can write to the memory by offset. This operator is only used to indicate that GE allocates contiguous memory and does not perform any computation.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input (dynamic) | List of input tensors. All inputs must have the same shape and data type. | ALL | ND |
| concat_dim | Required attribute | List of concatenation dimensions. The value range is `[-len(x.shape), len(x.shape)-1]`, which corresponds one-to-one with `N`. | LIST_INT | - |
| N | Required attribute | List of the numbers of concatenations on each concatenation dimension. The product of all elements must be equal to the number of input tensors, and each element must be greater than or equal to 1. | LIST_INT | - |
| keep_input_offset | Optional attribute | Whether to compute the memory offsets of input tensors. The default value is `true`. | BOOL | - |
| y | Output | Output tensor after concatenation. Its shape is the input shape multiplied by the corresponding `N` value on each `concat_dim` dimension, and its data type is the same as that of the input. | Same as `x` | ND |

## Constraint Description

- The number of elements in `concat_dim` must be consistent with that in `N`, and the product of all elements in `N` must be equal to the number of input tensors.
- All input tensors must have the same shape and data type.
- The memory size of input tensors must be 32-byte aligned (constraint of the host-side Host CPU execution engine).
- This operator does not support the compilation and execution of graphs with dynamic shapes (unknown shapes).
- The supported data types are subject to the IR definitions in [`phony_concat_proto.h`](op_graph/phony_concat_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_phony_concat](examples/test_geir_phony_concat.cpp) | Invokes the PhonyConcat operator by building a graph through the [operator IR](op_graph/phony_concat_proto.h). |
