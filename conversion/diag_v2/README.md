# DiagV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    ×     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |
| <term>Kirin X90处理器系列产品</term> |    √     |
| <term>Kirin 9030处理器系列产品</term> |    √     |

## 功能说明

- 算子功能：双向对角线算子（与numpy.diag语义一致）。

  如果输入是向量（一维向量），则返回二维矩阵张量，将输入元素填充到第diagonal条对角线上，其余位置为0；

  如果输入是二维张量，则输出一维向量，取值为diagonal指定的输入矩阵的对角线元素。

  如果diagonal = 0，选择主对角线；

  如果diagonal > 0，选择主对角线上方的第diagonal条对角线；

  如果diagonal < 0，选择主对角线下方的第-diagonal条对角线。

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
      <td>输入张量。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>可选属性</td>
      <td><ul><li>表示选择对角线的位置。</li><li>默认值为0，表示主对角线。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas训练系列产品、Atlas推理系列产品：不支持BFLOAT16。
- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16、COMPLEX。

## 约束说明

- 输入为一维或二维张量（rank 1/2），输出的数据类型和输入一样。
- 在输入为2维时，diagonal的取值应符合-(m-1) <= diagonal <= (n-1)，其中(m,n)为输入的shape。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_diag](./examples/test_aclnn_diag.cpp) | 通过[aclnnDiag](./docs/aclnnDiag.md)接口方式调用DiagV2算子。 |
| GE图模式 | [test_geir_diag_v2](./examples/test_geir_diag_v2.cpp) | 通过[算子IR](./op_graph/diag_v2_proto.h)构图方式调用DiagV2算子。 |
