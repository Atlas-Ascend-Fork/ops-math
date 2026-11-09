# Trace

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- |:----:|
| <term>Ascend 950PR&950DT系列产品</term>                             |  √   |
| <term>Atlas A3系列产品</term>     |  √   |
| <term>Atlas A2系列产品</term> |  √   |
| <term>Atlas 200I/500 A2推理产品</term>                      |  ×   |
| <term>Atlas推理系列产品</term>                             |  √   |
| <term>Atlas训练系列产品</term>                              |  √   |

## 功能说明

- 算子功能：
  Trace算子用于计算矩阵从左上角开始的主对角线元素的和。

- 计算公式：

  $$
  y = sum(diag(x))
  $$

## 参数说明

  <table style="table-layout: fixed; width: 1464px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 300px">
  <col style="width: 324px">
  <col style="width: 118px">
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
      <td>表示待计算对角线元素之和的二维张量，即公式中的x。</td>
      <td>COMPLEX64、FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示对角线元素之和，即公式中的y。UINT64输入对应UINT64输出，COMPLEX64、FLOAT、FLOAT16、BFLOAT16输入对应同类型输出，其余输入对应INT64输出。</td>
      <td>COMPLEX64、FLOAT、FLOAT16、BFLOAT16、INT64、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>：参数x和y的数据类型不支持BFLOAT16。

## 约束说明

- x必须为二维Tensor，y为零维标量。
- 对于形状为$(M, N)$的非方阵，参与累加的对角线长度为$\min(M, N)$。
- 支持形状为$(0, N)$、$(M, 0)$或$(0, 0)$的空Tensor，输出为0。

## 调用说明

| 调用方式   | 样例代码 | 说明  |
| ------------ | ------------ | ------------ |
| aclnn API  | [test_aclnn_trace](./examples/arch35/test_aclnn_trace.cpp) | 通过[aclnnTrace](./docs/aclnnTrace.md)接口调用Trace算子。   |
| GE图模式 | [test_geir_trace](./examples/test_geir_trace.cpp) | 通过[算子IR](./op_graph/trace_proto.h)构图调用Trace算子。 |
