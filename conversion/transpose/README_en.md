# Transpose

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |
| <term>Kirin X90 processor series products</term>                       |    √     |
| <term>Kirin 9030 processor series products</term>                       |    √     |

## Function Description

- Operator function: swaps arbitrary dimensions of a tensor. For example, if the input x is a tensor with shape [2, 3, 5] and perm is (2, 0, 1), the output y is a tensor with shape [5, 2, 3].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                                         | Data Type | Data Format |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | Input tensor       | Input tensor for dimension permutation.                                 | INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BFLOAT16, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4, HIFLOAT4_SCALE | ND       |
| perm   | Input tensor       | Represents the permutation of the dimensions of x. The number of elements must be equal to the number of dimensions of x. Each value must be in the range [-number of dimensions of x, number of dimensions of x - 1], and the elements must not be repeated. | INT32, INT64    | -        |
| y    | Output           | Has no more than 8 dimensions. The shape is jointly determined by perm and the shape of x (y.shape[i] = x.shape[perm[i]]), and the dtype must be the same as that of x. | Same as x   | ND       |

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4, and HIFLOAT4_SCALE are not supported.
- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: INT16, UINT16, UINT32, INT64, UINT64, BFLOAT16, BOOL, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT4, and HIFLOAT4_SCALE are not supported.

## Constraint Description

- The number of dimensions of x must not exceed 8.
- The number of elements in perm must be equal to the number of dimensions of x. Each value must be in the range [-number of dimensions of x, number of dimensions of x - 1], and the elements must not be repeated.
- The shape of y is jointly determined by perm and the shape of x, and the dtype must be the same as that of x.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_channel_shuffle](examples/test_aclnn_channel_shuffle.cpp) | Invokes the Transpose operator through the [aclnnChannelShuffle](docs/aclnnChannelShuffle_en.md) API. |
| aclnn API | [test_aclnn_permute](examples/test_aclnn_permute.cpp) | Invokes the Transpose operator through the [aclnnPermute](docs/aclnnPermute_en.md) API. |
| Graph mode | [test_geir_permute](examples/test_geir_permute.cpp) | Invokes the Transpose operator by building a graph with the [operator IR](op_graph/permute_proto.h). |
| Graph mode | [test_geir_transpose](examples/test_geir_transpose.cpp) | Invokes the Transpose operator by building a graph with the [operator IR](op_graph/transpose_proto.h). |
