# Expand

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>          |      √   |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×    |
| <term>Atlas inference products</term>                       |     √    |
| <term>Atlas training products</term>                       |     √    |

## Function Description

- Operator function: broadcasts the input tensor to the specified shape. For example, if the shape of the input tensor is (1, 4) and the specified shape is (2, 4), the output is a tensor with the shape (2, 4).

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                                         | Data Type | Data Format |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | Input tensor       | The tensor to be broadcast.  | FLOAT16, FLOAT, INT32, INT64, INT8, UINT8, BOOL, BF16   | ND       |
| shape   | Input tensor       | Indicates the shape size of x after broadcasting.| INT64, INT32    | -        |
| y    | Output           | The number of dimensions does not exceed 8. The shape is determined by the shape input, and the dtype must be the same as that of self. | Same as x   | ND       |

## Constraint Description

None

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Graph-mode invocation | [test_geir_expand](./examples/test_geir_expand.cpp) | Invokes the Expand operator through the GE IR graph mode |
| aclnn API | [test_aclnn_expand](./examples/test_aclnn_expand.cpp)   | Invokes the Expand operator through the [aclnnExpand](./docs/aclnnExpand_en.md) API. |
