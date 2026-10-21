# ControlTrigger

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT系列产品</term>       |   √  |
| <term>Atlas A3系列产品</term> |   √  |
| <term>Atlas A2系列产品</term> |   √  |
| <term>Atlas 200I/500 A2推理产品</term>          |   √  |
| <term>Atlas推理系列产品</term>                    |   √  |
| <term>Atlas训练系列产品</term>                    |   √  |

## 功能说明

- 算子功能：仅用作控制边的占位符。类似于NoOp，但即使部分控制输入已失效，也始终产生活跃的控制输出。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| 无 | - | ControlTrigger无输入、输出和属性。 | - | - |

## 约束说明

- ControlTrigger无数据输入和输出，仅支持作为控制边占位符使用。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_control_trigger](./examples/test_geir_control_trigger.cpp) | 通过[算子IR](./op_graph/control_trigger_proto.h)构图方式调用ControlTrigger算子。 |
