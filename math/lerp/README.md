# Lerp

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

- 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

- 计算公式：

$$
\text { out }_i=\text { start }_i+\text { weight }_i \times\left(\text { end }_i-\text { start }_i\right)
$$

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
      <td>start</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的start_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的end_i。数据类型与start保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>待进行lerp计算的入参，公式中的weight_i。数据类型与start保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行lerp计算的出参，公式中的out_i。数据类型与start保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

<term>Ascend 950PR&950DT系列产品</term>：start为float16或bfloat16时weight还支持float32。

## 调用说明

| 调用方式   | 样例代码                                       | 说明                                                                                 |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| aclnn API  | [test\_aclnn\_lerp](./examples/test_aclnn_lerp.cpp) <br> [test\_aclnn\_inplace\_lerp](./examples/test_aclnn_inplace_lerp.cpp) | 通过[aclnnLerp&aclnnInplaceLerp](./docs/aclnnLerp&aclnnInplaceLerp.md)接口方式调用Lerp算子。 |
| aclnn API  | [test\_aclnn\_lerps](./examples/test_aclnn_lerps.cpp) <br> [test\_aclnn\_inplace\_lerps](./examples/test_aclnn_inplace_lerps.cpp) | 通过[aclnnLerps&aclnnInplaceLerps](./docs/aclnnLerps&aclnnInplaceLerps.md)接口方式调用lerp算子。 |
| GE图模式 | [test\_geir\_lerp](./examples/test_geir_lerp.cpp)   | 通过[算子IR](./op_graph/lerp_proto.h)构图方式调用Lerp算子，参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)完成编译和验证。 |
