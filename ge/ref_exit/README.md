# RefExit

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

退出当前帧（frame）到其父帧。RefExit算子与Enter算子配合使用在图中创建循环，将可被引用覆盖的输入张量x传递到父帧使其可用。与Exit算子不同，RefExit作用于可被ref类算子覆盖的引用张量。输出张量y与输入张量x具有相同的形状和数据类型。

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
      <td>输入张量。需要传递到父帧的可被引用覆盖的张量。</td>
      <td>float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool</td>
      <td>ND</td>
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

- 输出张量y的数据类型和形状必须与输入张量x一致。

## 调用说明

RefExit算子为控制流算子，通常与Enter算子配合使用在图中创建循环，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_ref_exit](./examples/test_geir_ref_exit.cpp) | 通过[算子IR](./op_graph/ref_exit_proto.h)构图方式调用RefExit算子 |
