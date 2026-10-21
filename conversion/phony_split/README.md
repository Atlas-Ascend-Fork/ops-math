# PhonySplit

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

- 算子功能：将输入张量沿 `split_dim` 拆分为 `num_split` 个小张量。`PhonySplit` 的所有输出分配在同一块连续内存上，由GE计算每个输出的内存偏移，供后置自定义算子按偏移读取内存。该算子仅用于标识GE分配连续内存，不执行任何计算。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| split_dim | 必选属性 | 拆分维度列表，取值范围为 `[-len(x.shape), len(x.shape)-1]`，各维度值须能被对应的 `num_split` 整除。 | LIST_INT | - |
| num_split | 必选属性 | 每个拆分维度上的拆分数量列表，各元素大于等于1且乘积须等于输出张量个数。 | LIST_INT | - |
| keep_output_offset | 可选属性 | 是否计算输出张量的内存偏移，默认为 `true`。 | BOOL | - |
| y | 输出（动态） | 拆分后的输出张量列表，所有输出shape相同（输入shape在各 `split_dim` 维度上除以对应的 `num_split` 值），数据类型与输入一致。 | 与 `x` 相同 | ND |

## 约束说明

- `split_dim` 与 `num_split` 的元素个数须一致，`num_split` 各元素乘积须等于输出张量个数。
- 输入 `x` 在各 `split_dim` 维度上的值须能被对应的 `num_split` 整除。
- 所有输出张量的shape须一致（由InferShape保证）。
- 该算子不支持动态shape（unknown shape）图的编译执行。
- 支持的数据类型以 [`phony_split_proto.h`](op_graph/phony_split_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_phony_split](examples/test_geir_phony_split.cpp) | 通过[算子IR](op_graph/phony_split_proto.h)构图方式调用PhonySplit算子。 |
