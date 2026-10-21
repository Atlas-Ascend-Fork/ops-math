# PhonyConcat

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

- 算子功能：将一组张量沿 `concat_dim` 指定的维度拼接。`PhonyConcat` 的所有输入分配在同一块连续内存上，由GE计算每个输入的内存偏移，供前置自定义算子按偏移写入内存。该算子仅用于标识GE分配连续内存，不执行任何计算。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入（动态） | 输入张量列表，所有输入的shape和数据类型须一致。 | ALL | ND |
| concat_dim | 必选属性 | 拼接维度列表，取值范围为 `[-len(x.shape), len(x.shape)-1]`，与 `N` 一一对应。 | LIST_INT | - |
| N | 必选属性 | 每个拼接维度上的拼接数量列表，各元素乘积须等于输入张量个数且各元素大于等于1。 | LIST_INT | - |
| keep_input_offset | 可选属性 | 是否计算输入张量的内存偏移，默认为 `true`。 | BOOL | - |
| y | 输出 | 拼接后的输出张量，shape为输入shape在各 `concat_dim` 维度上乘以对应的 `N` 值，数据类型与输入一致。 | 与 `x` 相同 | ND |

## 约束说明

- `concat_dim` 与 `N` 的元素个数须一致，`N` 各元素乘积须等于输入张量个数。
- 所有输入张量的shape与数据类型须一致。
- 输入张量的内存大小须32字节对齐（主机侧Host CPU执行引擎约束）。
- 该算子不支持动态shape（unknown shape）图的编译执行。
- 支持的数据类型以 [`phony_concat_proto.h`](op_graph/phony_concat_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_phony_concat](examples/test_geir_phony_concat.cpp) | 通过[算子IR](op_graph/phony_concat_proto.h)构图方式调用PhonyConcat算子。 |
