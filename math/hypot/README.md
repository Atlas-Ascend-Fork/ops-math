# Hypot

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：逐元素计算两个输入张量对应元素构成的直角三角形斜边长度（二维欧几里得范数），输出恒为非负值。
- 计算公式：
  $$
  out_i = \sqrt{self_i^{2} + other_i^{2}}
  $$
- 数值稳定性：采用以绝对值较大一方为基准缩放的实现，避免中间平方项上溢或下溢；双零输入输出 `+0`，`inf` 优先于 `NaN`。

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
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入张量self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>公式中的输入张量other。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出张量out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- self 与 other 的 dtype 允许不一致，按互推导关系确定输出类型：任一侧为 FLOAT 或两侧 dtype 不同（FLOAT16 与 BFLOAT16）时输出 FLOAT，两侧同 dtype 时输出该 dtype。

## 约束说明

- 支持 0~8 维 ND Tensor，输入 shape 需满足 broadcast 关系。
- 支持空 Tensor 与非连续 Tensor。
- 不支持 DOUBLE、整型、BOOL 与复数类型。
- 本期仅提供 `aclnnHypot` 与图模式 `Hypot`，不提供原地（in-place）接口。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hypot](./examples/test_aclnn_hypot.cpp) | 通过[aclnnHypot](./docs/aclnnHypot.md)接口方式调用Hypot算子。    |
| 图模式调用 | [test_geir_hypot](./examples/test_geir_hypot.cpp)   | 通过[算子IR](./op_graph/hypot_proto.h)构图方式调用Hypot算子。 |
