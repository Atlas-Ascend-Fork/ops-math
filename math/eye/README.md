# Eye

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

算子功能：生成对角线上元素值为1、其余位置元素值为0的张量。未指定batch_shape时输出为二维(num_rows, num_columns)；指定batch_shape时输出为batch_shape + (num_rows, num_columns)的多维张量，每个二维切片对角线为1、其余为0。

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
      <td>num_rows</td>
      <td>属性</td>
      <td>表示输出张量的第一个维度，Host侧整型，取值范围大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_columns</td>
      <td>可选属性</td>
      <td>表示输出张量的第二个维度，Host侧整型，取值不大于0时表示与num_rows一致，默认值为0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_shape</td>
      <td>可选属性</td>
      <td>表示输出张量的batch维度，Host侧整型列表，每项必须大于0，默认值为[]（输出为二维）。</td>
      <td>INT64（ListInt）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>可选属性</td>
      <td>表示输出张量的数据类型，Host侧整型，对应数据类型枚举值，默认值为0（FLOAT32），支持的数据类型与y一致。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，对角线上元素值为1、其余位置元素值为0。shape为batch_shape + (num_rows, num_columns)，未指定batch_shape时为二维(num_rows, num_columns)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、UINT8、INT8、INT16、INT32、INT64、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_eye](./examples/test_aclnn_eye.cpp) | 通过[aclnnEye](./docs/aclnnEye.md)接口方式调用Eye算子，接口参数（n、m、out）及约束参见该文档。 |
| 图模式调用 | - | 图IR定义参见[Eye原型](./op_graph/eye_proto.h)。算子原型为`Eye(num_rows, num_columns=0, batch_shape=[], dtype=0) -> y`。 |
