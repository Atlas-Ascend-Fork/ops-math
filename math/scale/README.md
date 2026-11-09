# Scale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |   √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    √     |
| <term>Atlas训练系列产品</term>                              |   √     |

## 功能说明

- 算子功能：对输入Tensor进行scale和bias计算。若不输入bias，则 $y = x \cdot scale$；若输入bias，则 $y = x \cdot scale + bias$。
- 算子公式：

  若不输入bias，则

  $$
  y_i = x_i \cdot scale_i
  $$

  若输入bias，则

  $$
  y_i = x_i \cdot scale_i + bias_i
  $$

  - $x_i$为输入Tensor；
  - $scale_i$为缩放因子Tensor，支持与x的broadcast；
  - $bias_i$为可选的偏置Tensor，shape与scale保持一致；
  - $y_i$为输出Tensor，shape与x一致。

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
      <td>算子输入的Tensor。支持空Tensor。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>缩放因子Tensor。数据类型约束见下方约束说明，shape满足broadcast要求。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>偏置Tensor，默认不提供。数据类型约束见下方约束说明，shape要求见对应通路的接口说明。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出Tensor。shape必须与x完全一致，数据类型与x一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td>指定进行scale的起始轴。默认值为1，取值范围为[-x_rank, x_rank)。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_axes</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>指定进行scale的轴长度，默认值为1，取值范围大于等于-1。axis为负数时，按x_rank + axis归一化。</li>
          <li>当scale_from_blob为True时，-1表示缩放至末轴，0表示标量缩放；各通路的shape规则见对应接口说明。</li>
          <li>当scale_from_blob为False时，num_axes不参与scale的shape推导。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_from_blob</td>
      <td>可选属性</td>
      <td>取值为True或False，默认值为True。True按num_axes和axis推导scale shape；False按scale的rank和axis推导，num_axes不参与shape推导。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

表中属性默认值适用于GE图模式。aclnn接口的axis、numAxes、scaleFromBlob须显式传入，分别对应axis、num_axes、scale_from_blob。详细shape、动态形状和非连续Tensor约束见下方调用说明中的通路资料。

<term>Atlas推理系列产品</term>：aclnn API不支持BFLOAT16。

## 约束说明

- x与y的shape必须完全一致。
- 各输入dtype为FLOAT16/FLOAT/BFLOAT16之一，y的dtype与x一致。aclnn通路要求各输入dtype一致；GE图模式允许各输入分别选用这三种dtype，由GE适配后调用kernel。
- bias存在且非空Tensor时，bias与scale的shape必须一致；空Tensor的处理见对应通路的接口说明。
- scale的shape需满足axis和num_axes指定的broadcast规则，详见对应通路的接口说明。
- aclnn通路的x和scale维度不大于8；GE图模式的维度约束见[Scale IR定义](./op_graph/scale_proto.h)。
- axis取值范围为[-x_rank, x_rank)。
- num_axes取值范围大于等于-1。
- 仅支持ND格式。
- GE图模式支持-1未知维度和[-2]未知Rank描述符，运行时具体shape须满足[Scale IR定义](./op_graph/scale_proto.h)中的约束。
- aclnn通路支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，非连续Tensor的维度不大于8。

## 调用说明

| 调用方式 | 调用样例                                              | 说明                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| aclnn API | [test_aclnn_scale](./examples/test_aclnn_scale.cpp) | 通过[aclnnScale](./docs/aclnnScale.md)接口调用Scale算子。 |
| GE图模式 | - | 通过[Scale IR定义](./op_graph/scale_proto.h)构图，参数及动态形状约束见该定义。 |

<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas训练系列产品</term>支持GE图模式，不支持aclnn API。
