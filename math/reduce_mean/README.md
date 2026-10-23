# ReduceMean

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对一个多维向量按照指定的维度求平均值。
- 示例：
  - 示例1：
    定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对shape为(2, 3)的二维矩阵进行运算，对第一维计算数据求平均值，输出结果为[2.5, 3.5, 4.5]；对第二维计算数据求平均值，输出结果为[2, 5]。

    图1 ReduceMean按第一个维度计算示例
    ![alt text](./docs/fig/reduce_mean-by-dim-1.png)

    图2 ReduceMean最后一个维度计算示例
    ![alt text](./docs/fig/reduce_mean-by-dim-2.png)

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1034px">
<colgroup>
<col style="width: 82px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 352px">
<col style="width: 125px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>输入张量</td>
    <td>complex128, complex64, double, float32, float16, int64, int32, int16, int8 <br>
    uint64, uint32, uint16, uint8, bfloat16</td>
    <td>ND</td>
  </tr>
 <tr>
    <td>axes</td>
    <td>输入</td>
    <td>要缩减的维度。支持int、list、tuple或NoneType。数据类型必须为int32或int64。<br>如果为None，则缩减所有维度。取值范围为 [-rank(x), rank(x)]。</td>
    <td>int32, int64</td>
    <td>ND</td>
 </tr>
 <tr>
    <td>keep_dims</td>
    <td>属性</td>
    <td>可选布尔值，默认为false。若为true，则保留长度为1的缩减维度；<br>若为false，则对每个axis中的条目，张量的秩减1。</td>
    <td>bool</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>noop_with_empty_axes</td>
    <td>属性</td>
    <td>可选布尔值，默认为true。若为true，当axes = [] 时不进行缩减；<br> 若为false，当axes = [] 时缩减所有维度。</td>
    <td>bool</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出张量，与输入张量x的类型和格式相同</td>
    <td>与输入张量x相同</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- axes维度须大于0。
- 输入向量维度：最小1维，最大8维。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn调用 | [test_aclnn_mean_v2](./examples/test_aclnn_mean_v2.cpp) | 通过[aclnnMeanV2](./docs/aclnnMeanV2.md)接口方式调用ReduceMean算子。    |
| aclnn调用 | [test_aclnn_global_average_pool](./examples/test_aclnn_global_average_pool.cpp) | 通过[aclnnGlobalAveragePool](./docs/aclnnGlobalAveragePool.md)接口方式调用ReduceMean算子。    |
| 图模式调用 | [test_geir_reduce_mean](./examples/test_geir_reduce_mean.cpp)   | 通过[算子IR](./op_graph/reduce_mean_proto.h)构图方式调用ReduceMean算子。 |
