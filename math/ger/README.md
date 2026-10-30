# Ger

## 产品支持情况

| 产品                                         | 是否支持 |
| :------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>      |    √    |
| <term>Atlas A3系列产品</term>                |    √    |
| <term>Atlas A2系列产品</term>                |    √    |
| <term>Atlas 200I/500 A2推理产品</term>       |    ×    |
| <term>Atlas推理系列产品</term>               |    ×    |
| <term>Atlas训练系列产品</term>               |    √    |

## 功能说明

- 算子功能：对一维向量x1和x2计算外积（outer product），得到二维矩阵。
- 计算公式：
  $$
  y = x_1^T \times x_2
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                       | 数据类型               | 数据格式 |
| :--- | :--- | :----------------------------------------- | :--------------------- | :------- |
| x1   | 输入 | 公式中的输入向量x1，外积计算的第一个向量。 | BFLOAT16、FLOAT16、FLOAT | ND       |
| x2   | 输入 | 公式中的输入向量x2，外积计算的第二个向量。 | BFLOAT16、FLOAT16、FLOAT | ND       |
| y    | 输出 | 公式中的输出张量y，外积计算的结果，shape为{x1.shape[0]，x2.shape[0]}。 | BFLOAT16、FLOAT16、FLOAT | ND       |

- Atlas A3系列产品、Atlas A2系列产品、Atlas训练系列产品：不支持BFLOAT16数据类型。

## 约束说明

无

## 调用说明

| 调用方式   | 调用样例                                  | 说明                                             |
| :--- | :--- | :--- |
| aclnn API  | [test_aclnn_ger](./examples/test_aclnn_ger.cpp) | 通过[aclnnGer](./docs/aclnnGer.md)接口方式调用Ger算子。 |
| GE图模式   | -                                        | 通过[算子IR](./op_graph/ger_proto.h)构图方式调用Ger算子。 |
