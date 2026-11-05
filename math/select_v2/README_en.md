# SelectV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: selects and returns the corresponding elements from x1 or x2 based on the condition tensor (broadcast is supported). It is compatible with the SelectV2 operator of TensorFlow.
- Computation formula:

$$
y_i=\begin{cases}
  x1_i, & \text{if condition}_i \\
  x2_i, & \text{otherwise}
\end{cases}
$$

- Example:
  - If condition=[True, False], x1=[1, 2], and x2=[3, 4], the computation result is [1, 4].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                       | Data Type                           | Data Format |
| :--- | :--- | :----------------------------------------- | :--------------------------------- | :------- |
| condition | Input | Condition tensor, which is condition_i in the formula. When it is true, the corresponding element in x1 is selected. | BOOL | ND |
| x1 | Input | Input selected when the condition is true, which is x1_i in the formula. Its shape and the shapes of condition and x2 meet the broadcast relationship. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |
| x2 | Input | Input selected when the condition is false, which is x2_i in the formula. Its data type is the same as that of x1, and its shape and the shapes of condition and x1 meet the broadcast relationship. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |
| y | Output | Output tensor, which is y_i in the formula. Its shape is the shape of condition, x1, and x2 after broadcast, and its data type is the same as that of x1. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_s_where](../select/examples/test_aclnn_s_where.cpp) | Invokes the SelectV2 operator through the [aclnnSWhere](../select/docs/aclnnSWhere_en.md) API. The API is delivered by the select directory. |
| Graph-mode invocation | [test_geir_select_v2](./examples/test_geir_select_v2.cpp) | Invokes the SelectV2 operator by building a graph through the [operator IR](./op_graph/select_v2_proto.h). |
