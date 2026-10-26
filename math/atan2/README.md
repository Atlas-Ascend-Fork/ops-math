# Atan2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量self和other进行逐元素的反正切运算，self表示y坐标，other表示x坐标，计算结果为点(other, self)与x轴正方向的夹角（单位为弧度），取值范围为[-π, π]，支持广播。

- 计算公式：

  $$
  out_{i}=atan2(self_{i}, other_{i})
  $$

  其中，$self_{i}$表示输入张量self的第$i$个元素，$other_{i}$表示输入张量other的第$i$个元素。计算结果根据(other, self)所在象限确定：

  $$
  out_{i}=\begin{cases}
  atan(self_{i}/other_{i})+\pi & \text{if } other_{i}<0, self_{i}\ge 0 \\
  atan(self_{i}/other_{i})-\pi & \text{if } other_{i}<0, self_{i}<0 \\
  atan(self_{i}/other_{i}) & \text{if } other_{i}>0 \\
  \pi/2 & \text{if } other_{i}=0, self_{i}>0 \\
  -\pi/2 & \text{if } other_{i}=0, self_{i}<0 \\
  nan & \text{if } other_{i}=0, self_{i}=0
  \end{cases}
  $$

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
      <td>self</td>
      <td>输入</td>
      <td>待进行atan2计算的入参，公式中的self_i，表示y坐标。shape需与other满足broadcast关系。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>待进行atan2计算的入参，公式中的other_i，表示x坐标。shape需与self满足broadcast关系。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行atan2计算的出参，公式中的out_i。shape为self与other broadcast之后的shape。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- 输入self、other支持INT8、INT16、INT32、INT64、UINT8、BOOL类型，API内部会自动将其转换为浮点类型后参与计算。
- DOUBLE通过AICPU路径计算。
- Atlas推理系列产品、Atlas训练系列产品：不支持BFLOAT16。

## 约束说明

- self和other的shape需满足broadcast关系，out的shape为self与other broadcast之后的shape。
- 输入输出张量的维度不超过8，支持非连续Tensor。
- 确定性计算：aclnnAtan2&aclnnInplaceAtan2默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_atan2](./examples/test_aclnn_atan2.cpp) | 通过[aclnnAtan2和aclnnInplaceAtan2](./docs/aclnnAtan2&aclnnInplaceAtan2.md)接口方式调用Atan2算子。 |
