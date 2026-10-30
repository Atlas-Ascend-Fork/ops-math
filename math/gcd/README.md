# Gcd

## 产品支持情况

| 产品                                         | 是否支持 |
| :------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>      |    √    |
| <term>Atlas A3系列产品</term>                |    √    |
| <term>Atlas A2系列产品</term>                 |    √    |
| <term>Atlas 200I/500 A2推理产品</term>        |    ×    |
| <term>Atlas推理系列产品</term>                |    √    |
| <term>Atlas训练系列产品</term>                |    √    |

## 功能说明

- 算子功能：对两个输入张量逐元素计算最大公约数（greatest common divisor），输入需为整数类型。
- 计算公式：
  $$
  y = \gcd(x_1, x_2)
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                       | 数据类型                           | 数据格式 |
| :--- | :--- | :----------------------------------------- | :--------------------------------- | :------- |
| x1   | 输入 | 公式中的输入张量x1，shape与x2满足广播关系。 | UINT8、INT8、INT16、INT32、INT64  | ND       |
| x2   | 输入 | 公式中的输入张量x2，shape与x1满足广播关系。 | UINT8、INT8、INT16、INT32、INT64  | ND       |
| y    | 输出 | 公式中的输出张量y，shape为x1与x2广播后的shape。 | UINT8、INT8、INT16、INT32、INT64 | ND       |

- Atlas A3系列产品、Atlas A2系列产品、Atlas推理系列产品：不支持UINT8、INT8数据类型。
- Atlas训练系列产品：仅支持INT32数据类型。

## 约束说明

无

## 调用说明

| 调用方式   | 调用样例                                  | 说明                                             |
| :--- | :--- | :--- |
| aclnn API  | [test_aclnn_gcd](./examples/test_aclnn_gcd.cpp) | 通过[aclnnGcd](./docs/aclnnGcd.md)接口方式调用Gcd算子。 |
| GE图模式   | -                                        | 通过[算子IR](./op_graph/gcd_proto.h)构图方式调用Gcd算子。 |
