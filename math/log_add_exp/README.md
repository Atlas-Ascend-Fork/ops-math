# LogAddExp

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT系列产品</term>       |   √  |
| <term>Atlas A3系列产品</term> |   √  |
| <term>Atlas A2系列产品</term> |   √  |
| <term>Atlas 200I/500 A2推理产品</term>          |   ×  |
| <term>Atlas推理系列产品</term>                    |   √  |
| <term>Atlas训练系列产品</term>                    |   √  |

## 功能说明

- 算子功能：对两个输入张量逐元素计算指数和的对数，默认属性下以e为底。
- 计算公式：
  $$
  y = \ln\left(e^{x_1} + e^{x_2}\right)
  $$

## 参数说明

<table style="table-layout: fixed; width: 880px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 320px">
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
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入张量x_1</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入张量x2</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>base</td>
      <td>可选属性</td>
      <td>指数和对数运算的底数，默认值为-1.0，表示以e为底</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选属性</td>
      <td>指数项中-abs(x1-x2)的缩放系数，默认值为1.0</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>可选属性</td>
      <td>指数项的偏移量，默认值为0.0</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y</td>
      <td>同x1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- base必须为-1.0或大于0的值。
- x1、x2、y的数据类型需相同。
- x1与x2的shape需满足broadcast关系，y的shape为broadcast后的shape。
- <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>不支持BFLOAT16数据类型。

## 调用说明

| 调用方式    | 调用样例                                              | 说明                                                                           |
| ------- | ------------------------------------------------- | ---------------------------------------------------------------------------- |
| aclnn调用 | [test\_aclnn\_log\_add\_exp](./examples/test_aclnn_log_add_exp.cpp) | 通过[aclnnLogAddExp](./docs/aclnnLogAddExp.md)接口方式调用LogAddExp算子 |
| aclnn调用 | [test\_aclnn\_log\_add\_exp2](./examples/test_aclnn_log_add_exp2.cpp) | 通过[aclnnLogAddExp2](./docs/aclnnLogAddExp2.md)接口方式调用LogAddExp算子 |
