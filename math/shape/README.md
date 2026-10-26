# Shape

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：返回输入tensor的shape，即由输入各维度值组成的一维tensor。

- 计算逻辑：
  - 输入tensor的rank为r
  - 输出为一维tensor，长度为r，内容为输入tensor的各维度值

- 示例：

  ```text
  输入: x = float32[2, 3, 4]
  输出: y = int32[3] = [2, 3, 4]
  ```

## 参数说明

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 230px">
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
      <td>待获取shape信息的输入tensor。</td>
      <td>任意数据类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入tensor的各维度值组成的一维tensor，长度等于输入的rank。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输出tensor的数据类型，缺省时默认为INT32。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- dtype属性仅支持INT32、INT64，取其他值时算子推理失败。
- 输入为未知秩（-2）时，输出为一维未知形状（-1）。
- 输入仅支持ND格式。

## 调用说明

| 调用方式 | 调用样例                                              | 说明                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| 图模式调用 | [test_geir_shape](./examples/test_geir_shape.cpp)   | 通过[算子IR](./op_graph/shape_proto.h)构图方式调用Shape算子。                   |
