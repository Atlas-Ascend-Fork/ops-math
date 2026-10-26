# AippData

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

用于动态AIPP（AI Pre-Processing）场景。当AIPP配置文件中aipp-mode设置为dynamic时，框架会自动在图末尾添加一个AippData节点。AippData算子将输入张量data直接传递给输出张量out，同时通过index属性标识AIPP的序号。

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
      <td>输入张量，动态AIPP场景下通常为NCHW或NHWC排布的uint8类型数据。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>属性</td>
      <td>AIPP序号。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>与输入张量data具有相同形状和数据类型的张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 输出张量out的形状和数据类型必须与输入张量data一致。

## 调用说明

AippData算子由框架在动态AIPP场景下自动插入，不单独调用。

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_aipp_data](./examples/test_geir_aipp_data.cpp) | 通过[算子IR](./op_graph/aipp_data_proto.h)构图方式调用AippData算子。 |
