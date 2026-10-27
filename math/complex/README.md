# Complex

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

- 算子功能：根据实部real和虚部imag构建复数张量。

- 计算公式：

  $$
  \text{out}[i] = \text{real}[i] + \text{imag}[i] * \mathrm{j}
  $$

  其中，$\mathrm{j}$为虚数单位，real和imag按broadcast后的shape参与计算。

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
      <td>real</td>
      <td>输入</td>
      <td>复数的实部，公式中的real。</td>
      <td>FLOAT、FLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>imag</td>
      <td>输入</td>
      <td>复数的虚部，公式中的imag。</td>
      <td>FLOAT、FLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>复数张量，公式中的out。</td>
      <td>COMPLEX64、COMPLEX32、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- real和imag的数据类型必须一致，维度均不能超过8维，shape需满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)。
- out的shape必须是real与imag broadcast之后的shape。
- 在<term>Atlas A3系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>上，当输入为FLOAT、FLOAT16、DOUBLE时，输出数据类型分别应为COMPLEX64、COMPLEX32、COMPLEX128。
- real、imag和out均支持非连续Tensor。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_complex](./examples/test_aclnn_complex.cpp) | 通过[aclnnComplex](./docs/aclnnComplex.md)接口方式调用Complex算子。 |
