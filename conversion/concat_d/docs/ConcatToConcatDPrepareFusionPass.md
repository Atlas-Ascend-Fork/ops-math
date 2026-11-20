# ConcatToConcatDPrepareFusionPass

## 融合模式

该融合规则于图shape推导（infershape）前的kBeforeInferShape阶段执行，与[ConcatToConcatDFusionPass](ConcatToConcatDFusionPass.md)共享改写逻辑，仅匹配Concat算子，将其改写为ConcatD算子：

- `concat_dim`从第0个输入（常量）移入ConcatD的`concat_dim`属性。
- 数据输入一一对应连接到ConcatD输入；shape含0维的空张量输入在改写时被剪枝；数据输入超过512个时拆分为多级ConcatD树。
- 改写后ConcatD输出的dtype、shape、format与原算子输出保持一致。

融合前后图结构如下：

![](../../../docs/zh/figures/ConcatToConcatDPrepareFusionPass_1.png)

本融合规则在图编译流水线中的执行时序如下图所示：

![](../../../docs/zh/figures/ConcatToConcatDPrepareFusionPass_2.png)

ConcatV2、ConcatV2D算子不在本融合规则的处理范围内，由[ConcatToConcatDFusionPass](ConcatToConcatDFusionPass.md)改写。

## 使用约束

- 结构约束：
  - 图中存在Concat算子，`concat_dim`为第0个输入且必须是常量，数据类型为int32或int64。
  - 数据输入的数据类型需在ConcatD支持范围内（bfloat16、float16、float32、float64、int8/int16/int32/int64、uint8/uint16/uint32/uint64、bool、complex64、hifloat8、float8系列、float4系列）。
- 其余约束与ConcatToConcatDFusionPass一致。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
