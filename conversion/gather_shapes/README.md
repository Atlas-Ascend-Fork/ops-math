# GatherShapes

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

- 算子功能：从多个输入tensor的shape中收集指定的维度值，按axes顺序拼接为一维tensor输出。

- 计算逻辑：
  - axes中每个元素为二元组 [input_index, dim_index]，表示取第input_index个输入的第dim_index个维度值
  - 输出为一维tensor，长度等于axes的元素个数，内容为axes收集到的各维度值

- 示例：

  ```text
  输入: x0 = float32[2, 3], x1 = float32[4, 5, 6], axes = [[0,0],[0,1],[1,2]]
  输出: shape = int32[3] = [2, 3, 6]
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
      <td>动态输入</td>
      <td>待收集shape信息的输入tensor列表。</td>
      <td>任意数据类型</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>输出</td>
      <td>axes收集到的各输入维度值组成的一维tensor，长度等于axes的元素个数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>属性</td>
      <td>必选。二元组列表，每个元素 [input_index, dim_index]指定收集第input_index个输入的第dim_index个维度值。</td>
      <td>ListListInt（int64二元组列表）</td>
      <td>-</td>
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

- axes不能为空，且每个元素必须为二元组，input_index与dim_index不能越界（存在未知秩输入时跳过该检查）。
- dtype属性仅支持INT32、INT64，取其他值时算子推理失败。
- 输入仅支持ND格式。

## 调用说明

| 调用方式 | 调用样例                                              | 说明                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| 图模式调用 | [test_geir_gather_shapes](./examples/test_geir_gather_shapes.cpp)   | 通过[算子IR](./op_graph/gather_shapes_proto.h)构图方式调用GatherShapes算子。                   |
