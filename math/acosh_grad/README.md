# AcoshGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       | √ |
| <term>Atlas A3系列产品</term>     | √ |
| <term>Atlas A2系列产品</term>     | √ |
| <term>Atlas 200I/500 A2推理产品</term>                      | √ |
| <term>Atlas推理系列产品</term>                              | √ |
| <term>Atlas训练系列产品</term>                              | √ |

## 功能说明

- 算子功能：计算Acosh（反双曲余弦）算子的反向梯度。
- 算子公式：

  $$
  z_i = dy_i \cdot \dfrac{1}{\sinh(y_i)}
  $$

  其中：
  - $y_i$为前向Acosh算子的输出张量，值域期望$\geq 0$；
  - $dy_i$为上游传入的梯度；
  - $\sinh$为双曲正弦函数；
  - $z_i$为对前向原始输入张量的梯度，等于上游梯度乘以$1/\sinh(y_i)$。

- y为0时，输出可能出现Inf或NaN。

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
      <td>y</td>
      <td>输入</td>
      <td>前向Acosh算子的输出张量。值域期望落在[0, +∞)。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游传入的梯度张量，元素数和dtype与y一致，shape可不同。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>对前向原始输入张量的梯度，shape与dtype与y一致。</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- y与dy的元素数和dtype必须相同，shape可不同；输出shape与y相同。
- 图模式中，y与dy的广播shape必须等于y的shape，两输入的元素数仍需相同。
- 支持ND格式。
- 支持动态shape（-1）和未知rank（-2）。
- 满足上述输入约束的空Tensor返回空结果。
- 算子不限制输入输出的维数。
- 直接调用Kernel时，输入需为连续Tensor。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_acosh_grad](./examples/test_geir_acosh_grad.cpp) | 通过[算子IR](./op_graph/acosh_grad_proto.h)构图调用AcoshGrad。 |
| 图模式调用（arch35） | [test_geir_acosh_grad](./examples/arch35/test_geir_acosh_grad.cpp) | arch35目录下的调用样例。 |

示例编译与运行方法参见[快速算子调用（GE图模式）](../../docs/zh/invocation/quick_op_invocation.md)。
