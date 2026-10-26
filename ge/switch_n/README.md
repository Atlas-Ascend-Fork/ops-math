# SwitchN

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

SwitchN为控制流基础算子，根据整型张量输入`pred_value`将输入张量`data`转发到对应的输出端口。`output`为动态输出，其中只有一个输出会收到`data`，且其形状与数据类型与输入`data`一致。该算子通常与[Merge](../merge/README.md)算子配合实现多路分支控制。

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
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pred_value</td>
      <td>输入</td>
      <td>int64张量，决定数据被转发到哪个输出端口。</td>
      <td>DT_INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>动态输出</td>
      <td>输出张量，其中一个会收到data，数据类型同data。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 输入`data`的数据类型必须属于`float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool`之一。
- 输入`pred_value`的数据类型必须为`DT_INT64`。

## 调用说明

SwitchN算子为控制流基础算子，通常作为图中多路分支选择节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_switch_n](./examples/test_geir_switch_n.cpp) | 通过[算子IR](./op_graph/switch_n_proto.h)构图方式调用SwitchN算子 |
