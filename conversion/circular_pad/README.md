# CircularPad

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    ×     |
| <term>Atlas推理系列产品</term>                       |    ×     |
| <term>Atlas训练系列产品</term>                       |    ×     |
| <term>Kirin X90处理器系列产品</term> | √ |
| <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：使用循环填充方式对输入张量进行填充操作。本算子为单一GEIR算子，通过paddings长度区分2d/3d模式。
- CircularPad2d：对输入张量的最后两维进行循环填充。
- CircularPad3d：对输入张量的最后三维进行循环填充。

## 参数说明

<table style="table-layout: fixed; width: 974px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 251px">
<col style="width: 96px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待填充的原输入数据。shape支持3-5维：2d模式填充最后2维，3d模式填充最后3维。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>paddings</td>
    <td>输入</td>
    <td>填充配置。长度为4时数值依次代表左右上下需要填充的值（2d模式）；长度为6时数值依次代表左右上下前后需要填充的值（3d模式）。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>填充后的输出结果，维度数与x一致。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- Kirin X90处理器系列产品、Kirin 9030处理器系列产品：不支持BFLOAT16。

## 约束说明

- Ascend 950PR&950DT系列产品：CircularPad正padding可等于校验所用的对应尺寸（图模式为负padding裁剪后的有效尺寸，ACLNN接口为输入维度）；超过该尺寸不支持多圈绕回。
- Atlas A3系列产品、Atlas A2系列产品：y的最后一维在不同类型下的大小需满足如下约束。
  - int8：(0, 98304)
  - float16/bfloat16：(0, 49152)
  - int32/float32：(0, 24576)
- Kirin X90处理器系列产品、Kirin 9030处理器系列产品：y的最后一维在不同类型下的大小需满足如下约束。
  - int8：(0, 98304)
  - float16/bfloat16：(0, 49152)
  - int32/float32：(0, 24576)
- 输入和输出的数据类型必须一致。
- 2d模式（paddings长度4）要求x维度数为3或4；3d模式（paddings长度6）要求x维度数为4或5。即paddings长度必须等于2×被填充维数（2d填充末2维，3d填充末3维）。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_circular_pad](./examples/test_aclnn_circular_pad.cpp) | 通过[aclnnCircularPad2d](docs/aclnnCircularPad2d.md)、[aclnnCircularPad3d](docs/aclnnCircularPad3d.md)接口调用CircularPad算子。 |
| 图模式调用 | [test_geir_circular_pad](./examples/test_geir_circular_pad.cpp) | 通过[算子IR](./op_graph/circular_pad_proto.h)构图方式调用CircularPad算子。 |
