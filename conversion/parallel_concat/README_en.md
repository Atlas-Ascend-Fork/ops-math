# ParallelConcat

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- Operator function: concatenates N same-shaped tensors whose first dimension has a size of 1 along the first dimension. Each tensor in the dynamic input values has the shape [1, d_1..d_k], the output output_data has the shape [N, d_1..d_k], and the content of row i is the flattened content of the i-th input. Data is moved bit by bit, and arbitrary bit patterns such as NaN/Inf/±0 remain unchanged.
- Computation formula:

  $$
  \text{output\_data}[i, d_1, \ldots, d_k] = \text{values}_i[0, d_1, \ldots, d_k], \quad i = 0, 1, \ldots, N-1
  $$

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|-----|-----------|----|---------|------|
| values | Input | Dynamic input, which is a list of tensors to be concatenated. There are N tensors in total, and each tensor has the shape [1, d_1..d_k], which is `values_i` in the formula. The size of the first dimension must be 1, and all tensors must have the same shape and data type. | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL | ND |
| output_data | Output | Concatenation result tensor, whose shape is [N, d_1..d_k], which is `output_data` in the formula. Its data type is the same as that of values. | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL | ND |
| shape | Attribute | Explicit shape of the output tensor, which is [N, d_1..d_k]. Required attribute. | LIST_INT | - |
| N | Attribute | Number of tensors in the dynamic input values. N≥1 and N==shape[0]. Required attribute. | INT | - |

## Constraint Description

- The size of the first dimension of each input tensor must be 1, and the rank range is [1,8] (a scalar input with rank 0 is invalid).
- All input tensors must have the same shape and data type, and attr shape[1:] must be consistent with input shape[1:].
- The shape of the output output_data must be equal to attr shape.
- The empty tensor scenario is supported (when any d_m is 0, the output shape is [N, 0]).

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| GE graph mode | [test_geir_parallel_concat](examples/arch35/test_geir_parallel_concat.cpp) | Invokes the ParallelConcat operator by building a graph through the [operator IR](op_graph/parallel_concat_proto.h). |
