# RefEnter

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

创建或查找子帧（child frame），并使可被引用覆盖的输入张量x在子帧内可用。RefEnter算子与Exit算子配合使用在图中创建循环，通过唯一的frame_name属性标识帧。当is_constant为true时，输出张量y在子帧内为常量；否则在子帧内可能被改变。与Enter算子不同，RefEnter的输入可被ref类算子覆盖，起到变量的作用。

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
      <td>输入</td>
      <td>输入张量。需要在子帧内可用的可被引用覆盖的张量。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>frame_name</td>
      <td>属性</td>
      <td>子帧的名称，通过该唯一名称标识帧。</td>
      <td>String，必填</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_constant</td>
      <td>属性</td>
      <td>若为true，输出在子帧内为常量；否则在子帧内可能被改变。</td>
      <td>Bool，必填</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>与输入张量x具有相同形状和数据类型的张量。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- frame_name和is_constant为必填属性。
- 输出张量y的数据类型和形状必须与输入张量x一致。

## 调用说明

RefEnter算子为控制流算子，通常与Exit算子配合使用在图中创建循环，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_ref_enter](./examples/test_geir_ref_enter.cpp) | 通过[算子IR](./op_graph/ref_enter_proto.h)构图方式调用RefEnter算子 |
