# Dawsn

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

- 算子功能：逐元素计算Dawson积分（Dawson's integral）$F(x)$。

- 计算公式：

$$F(x) = e^{-x^{2}} \int_{0}^{x} e^{t^{2}} \mathrm{d}t$$

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
      <td>Dawson积分计算的输入张量，即公式中x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Dawson积分计算的输出张量，即公式中F(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x的维度数（rank）范围为0~8，支持0维标量输入，超出范围将在框架层被拒绝。
- 仅支持ND存储格式，其他格式将在Host侧Tiling校验阶段被拒绝。
- 支持空Tensor（0元素）输入，输出为同shape的空Tensor。
- 输入元素总数不超过288230376151710720（约$2.88\times10^{17}$），超出时算子将报错拒绝。
- 支持动态shape（-1未知维度）与动态rank（-2未知rank）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---------|----------------------------------------------------|------|
| GE图模式 | [test_geir_dawsn](./examples/test_geir_dawsn.cpp) | 通过GE图模式调用Dawsn算子。 |
