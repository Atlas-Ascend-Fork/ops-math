# FeedsRepeat

## 产品支持情况

| 产品                                              | 是否支持 |
|:---| :------: |
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

  对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小；未填充部分（超过实际复制行数的部分）置0。

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
    <td>-</td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。

## 约束说明

- feeds的秩（维度数）范围为1～8，第0维N必须与feeds_repeat_times的元素个数相同，且N≥1（不支持空Tensor，即N=0）；feeds第1维起各维的乘积（每行元素个数）必须≥1。
- feeds与feeds_repeat_times的数据类型分别为FLOAT/BFLOAT16/FLOAT16与INT32/INT64，其他类型不支持。
- feeds_repeat_times的元素总数必须等于feeds的第0维N（其秩不限，通常为一维）。构图期其维度允许为动态维（-1）或未知标记（此时元素总数无法校验，校验顺延至执行阶段）；执行阶段其具体维度值必须为非负整数。
- feeds_repeat_times中的负值按0处理。
- 当feeds_repeat_times各元素之和大于output_feeds_size时，超出output_feeds_size的重复行会被丢弃（输出按output_feeds_size截断）。
- 属性output_feeds_size必须为非负整数；负值会在算子执行准备阶段（tiling）被拒绝。
- 支持动态shape：GEIR图模式下构图时输入维度可使用-1（动态维），执行时以运行时下发的具体shape为准；feeds为[-2]（未知Rank）时，输出y传播为未知Rank[-2]。执行阶段要求feeds与feeds_repeat_times的具体维度值非负。
- 说明：-1/-2标记仅在构图与shape推导阶段生效——InferShape对含标记的输入跳过数值校验（或按未知Rank传播），而执行阶段由tiling以运行时下发的具体shape完成校验，具体维度值非负才会放行执行。
- 算子实现过程中，feeds_repeat_times需搬入UB并做精度转换和累加，其字节数受片上UB容量限制：tiling按2倍字节占用加约8 KB预留与UB容量比较，超出即拒绝。当前实现下，feeds_repeat_times占用的字节数在Atlas A2/A3系列产品不能超过约92 KB，在Ascend 950PR&950DT系列产品不能超过约120 KB。
- 行偏移量（各行的复制起点与清零位置）由feeds_repeat_times经float32累加得到；当单个或前缀累计的重复次数超过2^24时，行偏移可能出现偏差，请避免使用超过该量级的重复次数。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE图模式 | [test_geir_feeds_repeat](./examples/test_geir_feeds_repeat.cpp) | 通过[feeds_repeat_proto](op_graph/feeds_repeat_proto.h)接口方式调用feeds_repeat算子。 |
