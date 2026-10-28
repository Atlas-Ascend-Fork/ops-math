# MaximumGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR&950DT系列产品</term> |√|
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：MaximumGrad 是 `maximum` 算子的反向梯度计算算子，计算前向传播输出对两个输入 x1、x2 的梯度。
- 计算公式：

$$
y1 = grads \times (x1 \geq x2)
$$

$$
y2 = grads \times (x1 < x2)
$$

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|grads|输入|梯度输入，表示对 maximum 前向输出 y 的梯度，公式中的 `grads`。|FLOAT16、FLOAT、BF16、INT32|ND|
|x1|输入|前向传播的第一个输入，公式中的 `x1`。|FLOAT16、FLOAT、BF16、INT32|ND|
|x2|输入|前向传播的第二个输入，公式中的 `x2`。|FLOAT16、FLOAT、BF16、INT32|ND|
|y1|输出|对 x1 的梯度，公式中的 `y1`。|FLOAT16、FLOAT、BF16、INT32|ND|
|y2|输出|对 x2 的梯度，公式中的 `y2`。|FLOAT16、FLOAT、BF16、INT32|ND|
|grad_x|可选属性|是否计算并输出 y1，默认值 `true`。|BOOL|-|
|grad_y|可选属性|是否计算并输出 y2，默认值 `true`。|BOOL|-|

## 约束说明

- 输入 `grads`、`x1`、`x2` 三者的 shape 须满足 broadcast 规则。
- 输出 `y1`、`y2` 的 shape 为 `grads`、`x1`、`x2` broadcast 之后的 shape。
- 数据类型 BF16 仅 Ascend 950PR&950DT 系列产品支持。
- 属性 `grad_x` 与 `grad_y` 不能同时为 `false`。

## 调用说明

| 调用方式 | 调用样例                                                      | 说明                                                      |
|--------------|-----------------------------------------------------------|---------------------------------------------------------|
| 图模式调用 | [test_geir_maximum_grad](./examples/arch35/test_geir_maximum_grad.cpp) | 通过[算子IR](./op_graph/maximum_grad_proto.h)构图方式调用MaximumGrad算子。 |
