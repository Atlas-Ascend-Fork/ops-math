# Switch

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

Switch为控制流基础算子，根据布尔标量输入`pred`将输入张量`data`转发到对应的输出端口：当`pred`为`true`时，`data`转发到`output_true`；否则转发到`output_false`。该算子的两个输出`output_true`、`output_false`的形状与数据类型均与输入`data`一致，通常与[Merge](../merge/README.md)算子配合实现分支控制。

## 参数说明

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>data</td>
      <td>输入</td>
      <td>待转发的张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pred</td>
      <td>输入</td>
      <td>布尔标量，决定数据被转发到哪个输出端口。</td>
      <td>DT_BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_false</td>
      <td>输出</td>
      <td>当pred为false时，data转发到该输出，形状与数据类型同data。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_true</td>
      <td>输出</td>
      <td>当pred为true时，data转发到该输出，形状与数据类型同data。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 输入`pred`必须为布尔标量（rank为0或未知rank），否则形状推导失败。
- 输入`pred`的数据类型必须为`DT_BOOL`，否则数据类型推导失败。

## 调用说明

Switch算子为控制流基础算子，通常作为图中分支选择节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_switch](./examples/test_geir_switch.cpp) | 通过[算子IR](./op_graph/switch_proto.h)构图方式调用Switch算子 |
