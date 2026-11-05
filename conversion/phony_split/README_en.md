# PhonySplit

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

- Operator function: splits the input tensor along `split_dim` into `num_split` small tensors. All outputs of `PhonySplit` are allocated on the same contiguous memory, and GE computes the memory offset of each output so that succeeding custom operators can read the memory by offset. This operator is only used to indicate that GE allocates contiguous memory and does not perform any computation.

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | ALL | ND |
| split_dim | Required attribute | List of split dimensions. The value range is `[-len(x.shape), len(x.shape)-1]`, and the value of each dimension must be integer-divided by the corresponding `num_split`. | LIST_INT | - |
| num_split | Required attribute | List of the numbers of splits on each split dimension. Each element must be greater than or equal to 1, and the product of all elements must be equal to the number of output tensors. | LIST_INT | - |
| keep_output_offset | Optional attribute | Whether to compute the memory offsets of output tensors. The default value is `true`. | BOOL | - |
| y | Output (dynamic) | List of output tensors after splitting. All outputs have the same shape (the input shape divided by the corresponding `num_split` value on each `split_dim` dimension), and the data type is the same as that of the input. | Same as `x` | ND |

## Constraint Description

- The number of elements in `split_dim` must be consistent with that in `num_split`, and the product of all elements in `num_split` must be equal to the number of output tensors.
- The value of input `x` on each `split_dim` dimension must be integer-divided by the corresponding `num_split`.
- All output tensors must have the same shape (ensured by InferShape).
- This operator does not support the compilation and execution of graphs with dynamic shapes (unknown shapes).
- The supported data types are subject to the IR definitions in [`phony_split_proto.h`](op_graph/phony_split_proto.h).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_phony_split](examples/test_geir_phony_split.cpp) | Invokes the PhonySplit operator by building a graph through the [operator IR](op_graph/phony_split_proto.h). |
