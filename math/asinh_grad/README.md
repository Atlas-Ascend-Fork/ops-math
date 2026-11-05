# AsinhGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算asinh_grad(y, dy)，即反双曲正弦函数的梯度。
- 计算公式：

  $$
  z = \frac{dy}{\cosh(y)}
  $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>输入</td>
    <td>前向asinh的输出张量，对应公式中y=asinh(x)。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td>上游传入的梯度张量，对应公式中dy。数据类型须与y完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>z</td>
    <td>输出</td>
    <td>输出的输入梯度张量，对应公式中z。数据类型须与y完全一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- y、dy、z 的数据类型和 shape 必须一致，不支持广播。
- 支持 rank 0～8；rank 0 表示标量。允许空 Tensor。
- 非连续输入由图框架按 OpDef 的 AutoContiguous 配置转换后执行。
- 本目录提供 GEIR 图模式实现，不交付 ACLNN 接口。
- TensorFlow Asinh 的标准自动微分可能产生 Cosh/Div 组合图。保留的同名自定义节点映射不表示标准 TensorFlow 会生成 AsinhGrad 节点。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_asinh_grad.cpp](examples/test_geir_asinh_grad.cpp) | 通过图模式调用 [AsinhGrad IR](op_graph/asinh_grad_proto.h)；示例支持 `fp32`、`fp16`、`bf16` 参数。 |
