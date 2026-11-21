# ReduceStd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR&950DT系列产品</term>                     |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>           |  ×   |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                             |    ×     |

## 功能说明

- 算子功能：计算指定维度(dim)的标准差，这个dim可以是单个维度，维度列表或者None。计算逻辑与ReduceStdV2保持一致。
- 计算公式：
  假设dim为 $i$，则对该维度进行计算。$N$为该维度的shape。取 $self_{i}$，求出该维度上的平均值 $\bar{x_{i}}$。

  $$
  y1 = \sqrt{\frac{1}{max(0, N - \delta)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  $$

  其中 `unbiased=true` 时 `\delta=1`（样本标准差，Bessel校正），`unbiased=false` 时 `\delta=0`（总体标准差）。

  当`keepdim = true`时，reduce后保留该维度，且输出shape中该维度值为1；当`keepdim = false`时，不保留。

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
      <td>待进行ReduceStd计算的输入Tensor，公式中的self。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>参与Reduce计算的维度，公式中的i。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unbiased</td>
      <td>输入</td>
      <td>是否使用Bessel校正。true时分母为N-1，false时分母为N。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>输入</td>
      <td>是否在输出张量中保留输入张量的维度。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>ReduceStd计算的标准差出参。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>ReduceStd计算的均值出参。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_reduce_std](./examples/test_geir_reduce_std.cpp) | 通过[算子IR](./op_graph/reduce_std_proto.h)构图方式调用ReduceStd算子。 |
