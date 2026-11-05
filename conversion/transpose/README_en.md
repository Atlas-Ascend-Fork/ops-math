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

- Operator function: swaps arbitrary dimensions of a tensor. For example, if the input self is a tensor with shape [2, 3, 5] and dims is (2, 0, 1), the output is a tensor with shape [5, 2, 3].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                                         | Data Type | Data Format |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | Input tensor       | Input tensor for dimension permutation.                                 | See below   | ND       |
| perm   | Input tensor       | Represents the permutation of the dimensions of x. The value must be in the range [0, number of dimensions of self - 1]. | INT64, INT32    | -        |
| y    | Output           | Has no more than 8 dimensions. The shape is jointly determined by dims and the shape of the original self, and the dtype must be the same as that of self. | Same as x   | ND       |

- <term>Data types supported by the GPU</term>: The supported data types are DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16, DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, and DT_FLOAT8_E4M3FN.
- <term>Data types supported by the CPU</term>: The supported data types are the same as those of the GPU.

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_channel_shuffle](examples/test_aclnn_channel_shuffle.cpp) | Invokes the transpose operator through the [aclnn_channel_shuffle](docs/aclnnChannelShuffle_en.md) API. |
| aclnn API | [test_aclnn_permute](examples/test_aclnn_permute.cpp) | Invokes the transpose operator through the [aclnn_permute](docs/aclnnPermute_en.md) API. |
