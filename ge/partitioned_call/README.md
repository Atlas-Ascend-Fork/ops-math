# PartitionedCall

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

- 算子功能：将输入张量`args`传递给子图`f`，并返回子图`f`的输出张量。子图`f`接收输入并返回另一组张量。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `args` | 输入（动态） | 输入张量，将传递给子图`f`，支持动态输入。 | 所有类型 | ND |
| `f` | 子图 | 接收`args`并返回另一组张量的子图。 | - | - |
| `config` | 属性 | 可选字符串，默认为空。 | String | - |
| `config_proto` | 属性 | 可选字符串，默认为空。 | String | - |
| `executor_type` | 属性 | 可选字符串，默认为空。 | String | - |
| `output` | 输出（动态） | 子图`f`返回的输出张量，支持动态输出。 | 所有类型 | ND |

## 约束说明

- `args`的数量必须与`output`的数量一致。
- 各`args`与对应`output`的数据类型需保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_partitioned_call](./examples/test_geir_partitioned_call.cpp) | 通过[算子IR](./op_graph/partitioned_call_proto.h)构图方式调用PartitionedCall算子。 |
