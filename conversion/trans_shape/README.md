# TransShape

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>                                 |    √     |
| <term>Atlas A2系列产品</term>                                 |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                        |    √     |
| <term>Atlas推理系列产品</term>                                |    √     |
| <term>Atlas训练系列产品</term>                                |    √     |

## 功能说明

- 算子功能：根据 `outShape` 属性改变输出的shape，不改变底层数据。GE在图准备阶段会将 `TransShape` 节点替换为 `MemcpyAsync` 节点执行数据拷贝。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| outShape | 可选属性 | 输出的shape，默认为空列表（标量）。 | LIST_INT | - |
| y | 输出 | 输出张量，shape等于 `outShape`，数据类型与 `x` 一致。 | 与 `x` 相同 | ND |

## 约束说明

- `outShape` 各维度值的乘积应与输入 `x` 的元素总数一致，否则运行时数据拷贝会因内存不匹配而失败。
- 支持的数据类型以 [`trans_shape_proto.h`](op_graph/trans_shape_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_trans_shape](examples/test_geir_trans_shape.cpp) | 通过[算子IR](op_graph/trans_shape_proto.h)构图方式调用TransShape算子。 |
