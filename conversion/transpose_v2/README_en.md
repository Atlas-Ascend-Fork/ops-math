# TransposeV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    ×     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: performs the dimension permutation (Permutation) operation on a tensor, and rearranges the dimensions of the input tensor in the specified order. For example, if the input self is a tensor with shape [2, 3, 5] and dims is (2, 0, 1), the output is a tensor with shape [5, 2, 3].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                                         | Data Type | Data Format |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| self   | Input tensor       | Input tensor for dimension permutation.                                 | See below   | ND       |
| dims   | Input array       | Integer array that represents the dimensions of the original tensor and specifies the new axis order. The value must be in the range [-number of dimensions of self, number of dimensions of self - 1]. | INT32, INT64    | -        |
| out    | Output           | Has no more than 8 dimensions. The shape is jointly determined by dims and the shape of the original self, and the dtype must be the same as that of self. | Same as self   | ND       |

- <term>Atlas A3 products</term>: The supported data types are FLOAT, FLOAT16, and BFLOAT16.
- <term>Kirin X90/Kirin 9030 processor series products</term>: The supported data types are FLOAT and FLOAT16.

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API | [test_aclnn_transpose_v2](./examples/test_aclnn_transpose_v2.cpp) | Invokes the transposeV2 operator through the [aclnnPermute](../transpose/docs/aclnnPermute_en.md) API. |
