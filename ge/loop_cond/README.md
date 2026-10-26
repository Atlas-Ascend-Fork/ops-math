# LoopCond

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

- 算子功能：表示while循环的终止条件。将布尔标量输入张量`x`原样转发到输出张量`y`，用于向Switch算子传递循环终止条件，与Enter、Exit、Switch、NextIteration等算子配合构成循环控制结构。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 布尔标量，循环终止条件。 | bool | ND |
| y | 输出 | 与输入`x`形状、数据类型和内容相同的张量。 | 与`x`相同 | ND |

## 约束说明

- 输入`x`必须为标量（rank为0，或未知rank场景下维度为-2）。
- 输入`x`的数据类型必须为bool。
- 不支持含-1的动态shape输入（-1表示rank为1，不满足标量约束，构图校验会失败）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_loop_cond](examples/test_geir_loop_cond.cpp) | 通过[算子IR](op_graph/loop_cond_proto.h)构图方式调用LoopCond算子。 |
