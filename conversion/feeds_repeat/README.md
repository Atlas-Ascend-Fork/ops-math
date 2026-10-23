# FeedsRepeat

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

- 算子功能：

  对于输入feeds，根据输入feeds_repeat_times,将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 885px"><colgroup>
<col style="width: 137.333333px">
<col style="width: 104.333333px">
<col style="width: 369.333333px">
<col style="width: 205.333333px">
<col style="width: 68.333333px">
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
    <td>feeds</td>
    <td>输入</td>
    <td>等待处理的对象。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>feeds_repeat_times</td>
    <td>输入</td>
    <td>描述对feeds每行的repeat次数。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>repeat完成后的张量，dtype与feeds保持一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_feeds_size</td>
    <td>属性</td>
    <td>输出的第0维的数值等于该数值。</td>
    <td>INT</td>
    <td></td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。

## 约束说明

在算子实现过程中，feeds_repeat_times需搬入ub并做精度转换和累加，故feeds_repeat_times取值在Atlas A2/A3系列产品不能超过48 kb, 在Ascend 950PR&950DT系列产品系列产品不能超过64 kb。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| geir接口 | [test_geir_feeds_repeat](./examples/test_geir_feeds_repeat.cpp) | 通过[feeds_repeat_proto](op_graph/feeds_repeat_proto.h)接口方式调用feeds_repeat算子。 |
