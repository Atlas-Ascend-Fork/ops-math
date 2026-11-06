# FresnelSin

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    ×     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：计算Fresnel正弦积分 S(x) = integral_0^x sin(pi*t^2/2) dt。

- 计算公式：

$$S(x) = \int_{0}^{x} \sin\left(\frac{\pi}{2} t^{2}\right) \mathrm{d}t$$

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待进行Fresnel正弦积分计算的入参，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Fresnel正弦积分计算的出参，公式中的S(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas训练系列产品、Atlas推理系列产品：不支持BFLOAT16。

## 约束说明

- 不支持DOUBLE（FP64）数据类型。
- x 的维度数（rank）范围为 0~8，支持 0 维标量输入。
- 支持空 Tensor（0 元素）输入：Host 侧 Tiling 短路直接返回，输出为同 shape 空 Tensor（自动处理）。
- 支持非连续 Tensor 输入（Host 侧自动处理）。
- 特殊输入值的输出语义：输入为NaN时，输出为NaN。
- 大输入截断：当|x|≥36974（含±Inf）时，输出截断为sign(x)·0.5；有限大x（|x|≥36974）的截断输出与积分公式期望值的最大差约为8.6e-6（FLOAT32计算精度下）。
- 输入元素总数不得超过$2^{32}-1$（UINT32_MAX），超出时在Host Tiling阶段被校验拒绝并返回失败。
- 动态 shape 支持情况：支持动态 shape 与动态 rank（-1 未知维度 / -2 未知 rank）；实际执行要求各维在 Host Tiling 前具体化为非负维度，任一维度为负（无论元素总数正负或为零）都会在 Host Tiling 阶段被逐维校验拒绝；元素总数溢出（int64 溢出或超出 UINT32_MAX）同样在 Host Tiling 阶段被校验拒绝。

### 异常校验

| 违反条件 | 校验阶段 | 错误表现 |
| :------- | :------- | :------- |
| 缺少必选输入或输出 | 图构建/编译参数校验 | `E80001` |
| x 的 rank 大于 8 | 图编译参数校验（先于 Host Tiling）；Host Tiling 保留完整范围校验 | rank 大于 8 时当前集成通路呈现 `E80012`；若进入 Host Tiling 则返回 `GRAPH_FAILED` 并记录 rank 越界日志 |
| 输入 dtype 不在支持集（FLOAT、FLOAT16、BFLOAT16）内 | Host Tiling | 返回 `GRAPH_FAILED`，当前集成通路呈现 `EZ0019` 并记录 dtype 校验日志 |
| 非ND存储格式（如NCHW、FRACTAL_NZ） | Host Tiling | 返回 `GRAPH_FAILED`，当前集成通路呈现 `EZ0017` 并记录 format 校验日志 |

> 说明：错误码由图编译或运行框架对校验失败进行封装，不同集成通路可能只返回`GRAPH_FAILED`并保留具体日志。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_fresnel_sin](./examples/test_geir_fresnel_sin.cpp) | 通过[算子IR](./op_graph/fresnel_sin_proto.h)构图方式调用FresnelSin算子。 |

> 说明：样例运行时必须传入参数`<dtype_index>`指定输入数据类型：0表示FLOAT，1表示FLOAT16，2表示BFLOAT16。基于ops-math包执行样例的命令为`bash build.sh --run_example fresnel_sin graph [--soc=${soc_version}]`，该命令不透传样例参数，可追加`--noexec`仅编译样例，再直接运行编译产物`test_geir_fresnel_sin`（位于build目录）并传参，如`./test_geir_fresnel_sin 0`。
