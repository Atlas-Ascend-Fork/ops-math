# Merge

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

Merge为控制流基础算子，将输入`x`中第一个变为可用的张量转发到输出`y`，并在`value_index`中给出该张量在输入中的索引。Merge通常与[Switch](../switch/README.md)算子配合实现分支汇聚：Switch按条件把数据分流到不同分支，Merge再把可用分支的数据汇聚回主流程。

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
      <td>x</td>
      <td>动态输入</td>
      <td>输入张量，其中一个会变为可用。各输入张量应使用相同的数据类型；形状推导会合并各输入的维度信息。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>可用的张量，数据类型与x一致。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_index</td>
      <td>输出</td>
      <td>标量，被选中的输入张量在输入列表中的索引。</td>
      <td>DT_INT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 输入`x`的个数必须大于等于1，否则形状推导失败。
- 各输入`x`的数据类型必须一致；数据类型推导会逐个检查输入类型，不一致时返回失败。输出`y`的数据类型与输入一致，`value_index`为`DT_INT32`。
- 若各输入`x`的维度数不一致或存在未知秩输入，则输出`y`推导为未知秩（`[-2]`），不因秩不一致而返回推导失败。
- 对于某一维度，若各输入取值不一致或存在未知维（-1），则该维推导为-1（未知）。

## 调用说明

Merge算子为控制流基础算子，通常作为图中分支汇聚节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_merge](./examples/test_geir_merge.cpp) | 通过[算子IR](./op_graph/merge_proto.h)构图方式调用Merge算子 |
