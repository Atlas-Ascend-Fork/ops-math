# NextIteration

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

- 算子功能：使输入在下一轮迭代中可用。将输入张量`x`原样传递到输出张量`y`，用于在while循环中将当前迭代的计算结果传递给下一轮迭代，与Enter、Exit、Switch、LoopCond等算子配合构成循环控制结构。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 需要在下一轮迭代中可用的张量。 | float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool | ND |
| y | 输出 | 与输入`x`形状、数据类型和内容相同的张量。 | 与`x`相同 | ND |

## 约束说明

- `x`和`y`的形状及数据类型保持一致。
- 该算子为while循环控制流算子，需在Enter、Merge、Switch、LoopCond构成的完整循环结构内使用；脱离循环结构单独构图时无法执行，调用样例仅验证构图与形状推导。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_next_iteration](examples/test_geir_next_iteration.cpp) | 通过[算子IR](op_graph/next_iteration_proto.h)构图方式调用NextIteration算子。 |
