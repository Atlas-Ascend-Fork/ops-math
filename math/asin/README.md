# Asin

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量的每个元素计算反正弦，输出与输入shape一致的张量。
- 计算公式：

  $$
  y_{i} = \arcsin(x_{i})
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
      <td>x</td>
      <td>输入</td>
      <td>待计算反正弦的输入张量，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>反正弦计算结果，shape与x一致，公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- aclnn接口支持的数据类型、类型转换规则和维度约束见[aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin.md)。

## 约束说明

- x和y的shape一致。
- 图模式（GEIR）调用时x和y的数据类型必须一致；Tiling校验到输入、输出数据类型不同时返回失败，aclnn接口支持的类型转换见[aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin.md)。
- aclnn接口层的dtype、类型转换、空Tensor和维度约束见[aclnnAsin&aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_asin](examples/test_aclnn_asin.cpp)、[test_aclnn_asin_fp16](examples/test_aclnn_asin_fp16.cpp) | 通过[aclnnAsin](docs/aclnnAsin&aclnnInplaceAsin.md)接口方式调用Asin算子。 |
| aclnn原地调用 | - | 通过[aclnnInplaceAsin](docs/aclnnAsin&aclnnInplaceAsin.md)接口方式调用Asin算子。 |
| 图模式调用 | [test_geir_asin](examples/test_geir_asin.cpp)、[test_geir_asin_const_folding](examples/test_geir_asin_const_folding.cpp) | 通过[算子IR](op_graph/asin_proto.h)构图方式调用Asin算子。 |
