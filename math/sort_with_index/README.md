# SortWithIndex

[English](./README_en.md) | 简体中文

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量 x 沿指定维度进行升序或降序排序，并按相同顺序重排输入索引 index，输出排序结果 y 和重排后的索引 sorted_index。

- 计算公式：

$$p = \operatorname{argsort}(x, \text{axis}, \text{descending}, \text{stable})$$

$$y = \operatorname{gather}(x, \text{axis}, p)$$

$$\text{sorted\_index} = \operatorname{gather}(\text{index}, \text{axis}, p)$$

其中，p 表示排序后各元素在输入张量 x 中的位置。

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
      <td>待进行排序的输入张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>待随 x 重排的输入索引张量，shape 与 x 相同，索引值不参与排序比较。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>指定排序的维度，默认为-1（最后一维）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>属性</td>
      <td>是否降序排序，true为降序，false为升序，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>属性</td>
      <td>是否稳定排序，true为稳定排序，false为非稳定排序，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入张量 x 排序后的结果，shape、数据类型和数据格式与 x 相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>输出</td>
      <td>输入索引 index 按 x 的排序顺序重排后的结果，shape 与 x 相同。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：x 和 y 的数据类型仅支持FLOAT16、BFLOAT16；index 和 sorted_index 的数据类型仅支持INT32。

## 约束说明

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - 仅支持最后一维排序，axis 取值为-1或输入张量的维度数减1。
  - 不支持标量和空张量，输入张量各维度的长度必须为正整数。
  - 最后一维的长度不超过2147483647，其余维度长度的乘积不超过4294967295。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_sort_with_index](./examples/test_geir_sort_with_index.cpp) | 通过[算子IR](../less/op_graph/less_proto.h)构图方式调用SortWithIndex算子。 |
