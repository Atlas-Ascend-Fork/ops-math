# MaximumMinimumGradFusionPass

## 融合模式

该融合将图中GreaterEqual/LessEqual + Select/SelectV2替换为MaximumGrad或MinimumGrad算子：

- 比较算子为GreaterEqual时替换为MaximumGrad算子，为LessEqual时替换为MinimumGrad算子。
- 按子图包含的Select分支区分输出形态：同时包含dx、dy两个分支时，`grad_x=true, grad_y=true`，输出y1、y2；仅包含dx分支时，`grad_x=true, grad_y=false`，仅输出y1；仅包含dy分支时，`grad_x=false, grad_y=true`，仅输出y2。

融合前后图结构如下：

![](../../../docs/zh/figures/MaximumMinimumGradFusionPass.png)

替换后MaximumGrad或MinimumGrad输出的shape为输入广播后shape的结果，dtype、format与grads输入保持一致。

## 使用约束

- 子图须为TensorFlow自动求导生成的Maximum/Minimum梯度形态，比较算子节点名须位于`Maximum_grad`/`Minimum_grad`或`gradient_tape`命名空间下。
- 输入的数据类型须一致，且仅支持float16、float32、bf16、int32。
- Select/SelectV2的zeros输入必须为取值全零的常量节点。
- GE编译器版本不低于9.0.0时融合生效。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
