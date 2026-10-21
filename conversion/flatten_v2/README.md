# FlattenV2

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

- 算子功能：将输入张量 `[axis, end_axis]` 范围内的维度展平为一个维度，`axis` 之前的维度和 `end_axis` 之后的维度在输出中保留，不改变底层数据。算子由图引擎在主机侧（Host CPU）执行，作为内存复用（reuse-input）算子处理，输出直接视图输入内存。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| axis | 可选属性 | 展平的起始轴，之前的维度保留在输出中，默认为1。 | INT | - |
| end_axis | 可选属性 | 展平的结束轴，之后的维度保留在输出中，默认为 -1（最后一个维度）。 | INT | - |
| y | 输出 | 展平后的输出张量，数据类型与 `x` 一致。 | 与 `x` 相同 | ND |

## 约束说明

- `axis` 和 `end_axis` 必须在输入的维度范围内，且 `axis <= end_axis`。
- 若 `[axis, end_axis]` 范围内存在未知维度（-1），展平后的维度为 -1。
- 该算子仅支持在图中作为节点使用。
- 支持的数据类型以 [`flatten_v2_proto.h`](op_graph/flatten_v2_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_flatten_v2](examples/test_geir_flatten_v2.cpp) | 通过[算子IR](op_graph/flatten_v2_proto.h)构图方式调用FlattenV2算子。 |
