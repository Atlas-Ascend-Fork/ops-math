# Spence

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

- 算子功能：对输入张量的每个元素计算Spence函数（dilogarithm，双对数函数）值后输出。

- 计算公式：

$$\text{Spence}(x) = \int_1^x \frac{\ln t}{1 - t} \, dt = \text{Li}_2(1 - x)$$

特殊值：

- $x < 0$ 时输出 NaN
- $x = 0$ 时输出 $\pi^2/6 \approx 1.6449$
- $x = 1$ 时输出 $0$
- $x = +\infty$ 时输出 NaN

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
      <td>Spence函数的自变量，对应公式中x。x &lt; 0的元素输出NaN。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Spence函数的计算结果，对应公式中Spence(x)。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- FLOAT16、BFLOAT16输入在kernel内部提升到FLOAT32计算，结果再转回原始数据类型。
- 输入维度数不超过8，超过时算子报错；支持标量输入（按含1个元素的Tensor处理）、空Tensor、动态Shape与动态Rank。
- 非连续输入由框架自动连续化后参与计算。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|------------------------------------------------------------------|
| GE图模式 | [test_geir_spence](./examples/test_geir_spence.cpp) | 通过GE图模式调用Spence算子，定义算子IR的原型文件见[spence_proto.h](./op_graph/spence_proto.h)。 |
