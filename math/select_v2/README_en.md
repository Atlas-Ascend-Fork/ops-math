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

- Operator function: selects and returns the corresponding elements from then or else based on the condition tensor (broadcast is supported). It is compatible with the SelectV2 operator of TensorFlow.
- Computation formula:

$$
result_i=\begin{cases}
  then_i, & \text{if condition}_i \\
  else_i, & \text{otherwise}
\end{cases}
$$

- Example:
  - If condition=[True, False], then=[1, 2], and else=[3, 4], the computation result is [1, 4].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description                                       | Data Type                           | Data Format |
| :--- | :--- | :----------------------------------------- | :--------------------------------- | :------- |
| condition | Input | Condition tensor, which is condition_i in the formula. When it is true, the corresponding element in then is selected. | BOOL | ND |
| then | Input | Input selected when the condition is true, which is then_i in the formula. Its shape and the shapes of condition and else meet the broadcast relationship. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |
| else | Input | Input selected when the condition is false, which is else_i in the formula. Its data type is the same as that of then, and its shape and the shapes of condition and then meet the broadcast relationship. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |
| result | Output | Output tensor, which is result_i in the formula. Its shape is the shape of condition, then, and else after broadcast, and its data type is the same as that of then. | UINT8, INT8, INT32, INT64, FLOAT, FLOAT16, BFLOAT16 | ND |

## Constraint Description

None

## Invocation Description

| Invocation Method | Sample Code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_s_where](../select/examples/test_aclnn_s_where.cpp) | Invokes the SelectV2 operator through the [aclnnSWhere](../select/docs/aclnnSWhere_en.md) API. The API is delivered by the select directory. |
| GE graph mode | [test_geir_select_v2](./examples/test_geir_select_v2.cpp) | Invokes the SelectV2 operator by building a graph through the [operator IR](./op_graph/select_v2_proto.h). For compilation and verification, see [Compile and Run Samples](../../docs/en/context/compile_and_run_sample.md). |
