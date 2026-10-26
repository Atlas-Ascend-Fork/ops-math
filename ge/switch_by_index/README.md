# SwitchByIndex

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将指定下标的输入数据写入指定寄存器，用于运行时label切换。SwitchByIndex接收int32下标张量`x1`与一维uint64张量`x2`，无输出张量。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x1 | 输入 | 下标张量，shape为 (1,)。 | int32 | ND |
| x2 | 输入 | 一维张量。 | uint64 | ND |

## 约束说明

- `x1`的shape必须为 (1,)，数据类型必须为int32。
- `x2`必须为一维张量，数据类型必须为uint64。
- 该算子无输出张量，仅在图中作为内部切换节点使用；暂不支持单独执行，调用样例仅验证构图与形状推导。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_switch_by_index](examples/test_geir_switch_by_index.cpp) | 通过[算子IR](op_graph/switch_by_index_proto.h)构图方式调用SwitchByIndex算子。 |
