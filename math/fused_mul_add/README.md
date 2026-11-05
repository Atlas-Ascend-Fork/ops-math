# FusedMulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将`Mul`、`Add`子图融合为单个算子，对三个输入按NumPy广播规则对齐后逐元素计算乘加。

- 计算公式：

  $$
  y = x_1 \times x_2 + x_3
  $$

## 参数说明

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的乘法输入张量x1。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的乘法输入张量x2，shape需与x1可广播。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>公式中的加法输入张量x3，shape需与x1*x2的结果可广播。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y，shape为x1、x2、x3广播后的统一形状。</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x1、x2、x3、y必须为同一种数据类型，不支持混合数据类型。
- x1、x2、x3的维度数（rank）均不能超过8，超过时在tiling阶段校验失败。
- 支持任意NumPy广播形态（含标量、单维broadcast、跨rank broadcast），支持动态shape与动态rank。
- <term>Ascend 950PR&950DT系列产品</term>：支持空Tensor（0元素）输入，输出为广播后的空Tensor（元素数为0）。

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_fused_mul_add](examples/arch35/test_geir_fused_mul_add.cpp) | 通过[算子IR](op_graph/fused_mul_add_proto.h)构图方式调用FusedMulAdd算子。 |
