# HistogramV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    ×     |

> 说明：BFLOAT16数据类型仅在Ascend 950PR&950DT系列产品（arch35/regbase）上支持。

## 功能说明

- 算子功能：计算张量直方图。
- 计算公式：以min和max作为统计上下限，在min和max之间划出等宽的数量为bins的区间，统计张量x中元素在各个区间的数量。如果min和max相等，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>输入</td>
      <td>公式中的输入张量x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>输入</td>
      <td>公式中的输入张量min，只含有1个元素。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>输入</td>
      <td>公式中的输入张量max，只含有1个元素。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，直方图的计算结果。</td>
      <td>FLOAT32、INT32（Ascend 950PR&950DT系列产品跟随输入x推导；其他产品由y_dtype选择）</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bins</td>
      <td>可选属性</td>
      <td><ul><li>表示被划分的区间数量。</li><li>默认值为100。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>可选属性</td>
      <td><ul><li>表示输出数据类型，取值为INT32或FLOAT（float32）。</li><li>Ascend 950PR&950DT系列产品：输出类型由输入x推导（浮点→FLOAT32，整数→INT32），无需设置该属性；显式设置时须与推导结果一致，否则报错。</li><li>其他产品：由该属性选择输出类型，默认值为INT32。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `bins`取值范围为(0, 2147483647]：大于0，且不超过INT32_MAX（kernel分箱下标为int32）。
- `min`不得大于`max`；`min == max`时先按张量元素的最小值/最大值重算统计上下限，重算后仍相等时按邻接可表示值向两侧展开。
- `min`/`max`需为有限值；仅允许两者同为+inf或同为-inf。
- Ascend 950PR&950DT系列产品（arch35）上：整数类型输入时`min`/`max`按int64语义解释；标量超出输入数据类型值域时，输入与标量会共同提升后再计算。其他产品保持既有语义。
- 输出类型：Ascend 950PR&950DT系列产品跟随输入x推导（浮点→FLOAT32，整数→INT32），无需设置`y_dtype`；其他产品由`y_dtype`选择（默认INT32）。
- 浮点输出默认使用原子累加（非确定性），可通过`aclrtSetSysParamOpt`开启确定性；整数/INT32输出为确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_histc](./examples/test_aclnn_histc.cpp) | 通过[aclnnHistc](./docs/aclnnHistc.md)接口方式调用HistogramV2算子。 |
| 图模式调用 | [test_geir_histogram_v2](./examples/test_geir_histogram_v2.cpp)   | 通过[算子IR](./op_graph/histogram_v2_proto.h)构图方式调用HistogramV2算子。 |
