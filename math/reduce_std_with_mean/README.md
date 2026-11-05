# ReduceStdWithMean

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

> **说明**:
>
> Ascend 950PR&950DT系列产品：由本自定义算子包提供kernel实现（arch35）；其余产品由CANN内置算子库支持。

## 功能说明

给定外部预计算的均值 `mean`，沿指定维度 `dim` 计算输入 `x` 的标准差（`invert=false`）或其倒数（`invert=true`）。是 BatchNorm/LayerNorm/GroupNorm 归一化层统计量计算的关键子步骤，以及 PyTorch `torch.std`/`torch.std_mean`/`torch.batch_norm_stats` 在 NPU 上的底层实现。

标准差模式（`invert=false`，`correction` 生效）：

$$
y = \sqrt{\frac{\sum_{j=0}^{N-1}(x_j - mean_j)^2}{\max(0,\; N - correction)}}
$$

标准差倒数模式（`invert=true`，固定除以 N、加 epsilon，忽略 correction）：

$$
y = \frac{1}{\sqrt{\dfrac{1}{N}\sum_{j=0}^{N-1}(x_j - mean_j)^2 + \epsilon}}
$$

其中 $N=\prod_{d \in dim}\text{shape}[d]$ 为归约维度元素总数，$correction$ 为 Bessel 校正因子（0=总体、1=样本），$\epsilon$ 为数值稳定小量。

ReduceStdWithMean 为 CANN 内部 L0 算子，被 `aclnnStd`、`aclnnStdMeanCorrection`、`aclnnBatchNormStats` 三个上层 aclnn 接口组合调用，不直接暴露独立 aclnn 接口。

| 上层接口 | 功能 | correction | invert | eps |
| :--- | :--- | :--- | :--- | :--- |
| aclnnStd | 独立标准差 | 用户传入 | false | — |
| aclnnStdMeanCorrection | 同时输出 (std, mean) | 用户传入 | false | 0.001（固定） |
| aclnnBatchNormStats | 输出 (mean, invstd) | 0（固定） | true | 用户传入 |

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>待计算标准差的张量。</td>
<td>FLOAT16、FLOAT、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mean</td>
<td>输入</td>
<td>外部预计算均值（已广播到x形状），dtype与x一致。</td>
<td>与x一致</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>输出张量，dtype与x一致；shape 由dim+keepdim推导。</td>
<td>与x一致</td>
<td>ND</td>
</tr>
<tr>
<td>dim</td>
<td>可选属性</td>
<td>归约维度，范围[-rank(x), rank(x)-1]，空表示全部维度。默认{}。</td>
<td>ListInt</td>
<td>-</td>
</tr>
<tr>
<td>unbiased</td>
<td>可选属性</td>
<td>Bessel 校正（legacy字段）。默认true。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>keepdim</td>
<td>可选属性</td>
<td>是否保留归约维为size 1。默认false。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>invert</td>
<td>可选属性</td>
<td>true 返回1/sqrt(var+eps)，false返回sqrt(var)。默认false。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>epsilon</td>
<td>可选属性</td>
<td>数值稳定项（仅invert=true使用），须为非负浮点数。默认0.001。</td>
<td>Float</td>
<td>-</td>
</tr>
<tr>
<td>correction</td>
<td>可选属性</td>
<td>校正因子：0=总体，1=样本。默认1。</td>
<td>Int</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- Atlas 200I/500 A2推理产品、Atlas推理系列产品、Atlas训练系列产品：不支持BFLOAT16数据类型。
- x的shape维度不超过 8 维。
- 支持rank=0标量输入（x/mean为0-D）：dim为空全归约，输出为标量（keepdim=true 时输出1-D size 1）。
- mean的shape须与x完全一致（mean须已广播到x形状）。
- dim取值范围 [-rank(x), rank(x)-1]，空表示全部维度归约，且不能包含重复轴（负数轴归一化后判重）。
- 输出y的shape须与`dim+keepdim`推导结果一致。
- correction仅支持非负整数；invert=true时忽略correction，固定除以N。
- epsilon仅支持非负浮点数（默认0.001），tiling阶段对`epsilon<0`返回错误。
- unbiased为legacy字段：correction属性有值时以correction为准，无值时按`unbiased=true→correction=1`、`unbiased=false→correction=0`折算。
- 空张量（N=0）输出NaN；N<=correction（invert=false）输出+Inf或NaN（一般输入为+Inf，全等输入为0×Inf=NaN）。
- 输入含动态标记（-1/-2）时按动态shape处理，具体语义由下游框架定义。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_reduce_std_with_mean.cpp](examples/test_geir_reduce_std_with_mean.cpp) | 通过 GE IR 构图调用，算子 IR 原型见 [op_graph/reduce_std_with_mean_proto.h](op_graph/reduce_std_with_mean_proto.h) |
| aclnn API | [test_aclnn_std_mean_correction.cpp](examples/test_aclnn_std_mean_correction.cpp) | 通过 [aclnnStdMeanCorrection](docs/aclnnStdMeanCorrection.md) 接口方式调用ReduceStdWithMean算子。 |
| aclnn API | [test_aclnn_batch_norm_stats.cpp](examples/test_aclnn_batch_norm_stats.cpp) | 通过 [aclnnBatchNormStats](docs/aclnnBatchNormStats.md) 接口方式调用ReduceStdWithMean算子。 |
| aclnn API | [test_aclnn_std.cpp](../reduce_std_v2/examples/test_aclnn_std.cpp) | 通过 [aclnnStd](../reduce_std_v2/docs/aclnnStd.md) 接口方式调用ReduceStdWithMean算子（该接口实现在同仓 `math/reduce_std_v2`）。 |
