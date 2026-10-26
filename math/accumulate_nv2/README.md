# AccumulateNV2

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    √     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：返回输入tensors列表中每个输入tensor依次做add求和。AccumulateNv2与AddN执行相同的运算，但不等待所有输入就绪后再开始求和，可在输入就绪时间不同时节省内存。

- 计算公式：

$$y = \sum_{i=0}^{N-1} x_i$$

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
      <td>需要求和的输入tensor列表，公式中的x_i。图模式直连时各输入shape需完全相同（算子本体不支持broadcast）；通过aclnnSum接口调用时各tensor的shape可不同，由接口层完成对齐。</td>
      <td>FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>求和结果tensor，公式中的y。</td>
      <td>FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>N</td>
      <td>属性</td>
      <td>输入tensor列表的大小。N需大于等于1，且必须与实际输入Tensor的数量一致；不一致时在图编译阶段报错。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 通过aclnnSum接口调用时，输入tensors列表中各tensor的shape需满足broadcast关系，由实现层完成对齐；通过图模式（GE/GEIR）直连算子时，各输入shape需完全相同且与输出y一致（算子本体不支持broadcast），shape不一致时在图编译阶段报错。
- 输入tensors与输出的数据类型需相同，且tensors列表内各tensor的数据类型需一致；不一致时在图编译阶段报错。
- 输入维度不超过8维：aclnnSum接口第一段校验拦截（ACLNN_ERR_PARAM_INVALID 161002）；图模式直连时在tiling阶段拦截。
- 图模式（GE/GEIR）直连时输入数量需在[1, 40]范围内（tiling阶段拦截，超限报错）；aclnnSum接口内部按16个一组分组累加，不受该上限约束。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sum](./examples/test_aclnn_sum.cpp) | 通过[aclnnSum](./docs/aclnnSum.md)接口方式调用AccumulateNV2算子。 |
| 图模式（GEIR） | [accumulate_nv2_proto.h](./op_graph/accumulate_nv2_proto.h) | 通过[AccumulateNV2算子IR定义](./op_graph/accumulate_nv2_proto.h)在GE图中调用（各输入shape需完全相同）。 |
