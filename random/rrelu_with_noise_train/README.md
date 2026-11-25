# RreluWithNoiseTrain

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                      |    √     |
| <term>Atlas A3系列产品</term>                                |    ×     |
| <term>Atlas A2系列产品</term>                                |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                       |    ×     |
| <term>Atlas推理系列产品</term>                                |    ×     |
| <term>Atlas训练系列产品</term>                                |    ×     |

## 功能说明

- 算子功能：RReLU是leaky ReLU的随机化变体：训练时对输入的每个非正元素，从均匀分布U(lower,upper)中采样一个斜率因子a。本算子在输出激活结果y的同时，将每个位置实际使用的斜率记录到输出noise中供反向传播等后续使用。
- 计算公式（training语义）：

  $$
  y[i] =
  \begin{cases}
  x[i] \cdot r[i], & x[i] \leq 0 \\
  x[i], & x[i] > 0
  \end{cases}
  \qquad
  noise[i] =
  \begin{cases}
  r[i], & x[i] \leq 0 \\
  1, & x[i] > 0
  \end{cases}
  $$

- 其中 $r[i]$为$[lower, upper]$区间上的均匀分布随机数（浮点舍入可能触及上界），由seed与offset共同决定的确定性随机数流生成，与数据无关：$r[i] = (T)u[i] \cdot (upper - lower) + lower$，$u[i] \in (0, 1]$。
- 分支语义：$x[i] = 0$走随机斜率分支；$x[i]$为NaN时比较恒为假，走正分支（$y[i] = NaN$，$noise[i] = 1$）。
- 同seed与offset下随机流结果逐次确定。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>输入tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor，shape与dtype与x一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>输出</td>
      <td>随机斜率噪声tensor：x≤0位置为对应随机斜率r，x&gt;0位置为1。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lower</td>
      <td>属性</td>
      <td>随机斜率采样区间下界，默认0.125。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>upper</td>
      <td>属性</td>
      <td>随机斜率采样区间上界，默认0.333333。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数的种子，影响生成的随机数序列。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数的偏移量，它影响生成的随机数序列的位置。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- offset须为4的倍数

## 调用说明

| 调用方式     | 样例代码                                                     | 说明                                                         |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | - | 通过[aclnnRReluWithNoise](https://gitcode.com/cann/ops-nn/blob/master/activation/leaky_relu/docs/aclnnRReluWithNoise%26aclnnInplaceRReluWithNoise.md)接口方式调用RreluWithNoiseTrain算子。 |
