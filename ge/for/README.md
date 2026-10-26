# For

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

- 算子功能：循环执行`body`子图，循环变量从`start`开始，每次迭代递增`delta`，直到循环变量超过`limit`。每次迭代时，将`input`传入`body`子图，`body`返回的输出作为下一次迭代的输入，最终输出为最后一次迭代的结果。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `start` | 输入 | 循环下界，int32标量。 | INT32 | ND |
| `limit` | 输入 | 循环上界，int32标量。 | INT32 | ND |
| `delta` | 输入 | 步长大小，int32标量。 | INT32 | ND |
| `input` | 输入（动态） | 输入张量，支持动态输入。 | 所有类型 | ND |
| `body` | 子图 | 接收`input`并返回另一组张量的子图，用于计算下一次迭代的值。 | - | - |
| `output` | 输出（动态） | `body`子图返回的输出张量，类型与`input`相同，支持动态输出。 | 所有类型 | ND |

## 约束说明

- `start`、`limit`、`delta`必须为INT32标量。
- `input`的数量必须与`output`的数量一致。
- 各`input`与对应`output`的数据类型需保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_for](./examples/test_geir_for.cpp) | 通过[算子IR](./op_graph/for_proto.h)构图方式调用For算子。 |
