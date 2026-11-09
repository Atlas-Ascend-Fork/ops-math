# Ndtri

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

- 算子功能：逆标准正态累积分布函数（probit / inverse normal CDF）。对输入概率张量逐元素计算标准正态分位点，底层采用Cephes数学库的分区间有理逼近算法。

- 计算公式：

$$y_i = \mathrm{ndtri}(x_i) = \Phi^{-1}(x_i) = \sqrt{2} \cdot \mathrm{erf}^{-1}(2 x_i - 1), \quad x_i \in (0, 1)$$

- 特殊值处理：

| 输入 $x_i$ | 输出 $y_i$ |
| :---: | :---: |
| `0` | `-inf` |
| `1` | `+inf` |
| FP32、BF16正、负次正规数 | `-inf`，计算时将输入视为零 |
| $x_i < 0$（上述次正规数除外）或 $x_i > 1$ | `NaN` |
| `NaN` / `+inf` / `-inf` | `NaN` |

FP16输入先升精度为FP32，其次正规数在FP32中为正规数，按原数值算法计算。

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
      <td>待进行ndtri计算的概率张量，公式中的x_i，推荐值域(0, 1)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>逆标准正态CDF的计算结果，公式中的y_i。输出shape、dtype与输入一致；标量输出仍为标量；空Tensor返回同形状的空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x的维度不超过8维。
- 运行时输入x的各维度大小必须为非负整数，元素总数不得超过INT64类型的最大值。

## 调用说明

| 调用方式 | 样例代码                                       | 说明                                                              |
|--------------|--------------------------------------------|-----------------------------------------------------------------|
| GE图模式 | [test_geir_ndtri](./examples/test_geir_ndtri.cpp) | 通过[算子IR](./op_graph/ndtri_proto.h)构图方式调用Ndtri算子。 |
