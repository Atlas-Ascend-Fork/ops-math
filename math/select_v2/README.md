# SelectV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：根据条件张量condition选取x1或x2中对应元素并返回（支持广播），兼容TensorFlow的SelectV2算子。
- 计算公式：

$$
y_i=\begin{cases}
  x1_i, & \text{if condition}_i \\
  x2_i, & \text{otherwise}
\end{cases}
$$

- 示例：
  - 若condition=[True, False]，x1=[1, 2]，x2=[3, 4]，计算结果为[1, 4]。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                       | 数据类型                           | 数据格式 |
| :--- | :--- | :----------------------------------------- | :--------------------------------- | :------- |
| condition | 输入 | 条件张量，公式中的condition_i，为真时选取x1中对应元素。 | BOOL | ND |
| x1 | 输入 | 条件为真时选取的输入，公式中的x1_i，shape与condition、x2满足广播关系。 | UINT8、INT8、INT32、INT64、FLOAT、FLOAT16、BFLOAT16 | ND |
| x2 | 输入 | 条件为假时选取的输入，公式中的x2_i，数据类型与x1保持一致，shape与condition、x1满足广播关系。 | UINT8、INT8、INT32、INT64、FLOAT、FLOAT16、BFLOAT16 | ND |
| y | 输出 | 输出张量，公式中的y_i，shape为condition、x1、x2广播后的shape，数据类型与x1保持一致。 | UINT8、INT8、INT32、INT64、FLOAT、FLOAT16、BFLOAT16 | ND |

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_s_where](../select/examples/test_aclnn_s_where.cpp) | 通过[aclnnSWhere](../select/docs/aclnnSWhere.md)接口方式调用SelectV2算子，接口由select目录交付。 |
| 图模式调用 | [test_geir_select_v2](./examples/test_geir_select_v2.cpp) | 通过[算子IR](./op_graph/select_v2_proto.h)构图方式调用SelectV2算子。 |
