# StatelessUniform

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    ×     |
| <term>Atlas A2系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：基于Philox4x32-10伪随机数生成算法，生成服从均匀分布的随机数张量，随机值范围为[from, to)，边界与上界回绕按转换为输出数据类型后的值解释。

- 计算公式：

  记$T$为输出数据类型（FLOAT32、FLOAT16或BFLOAT16），$from$、$to$为DOUBLE标量，在tiling阶段先转换为FLOAT32，再分别转换为$T$得到边界$a$、$b$，差值$d$同样按$T$保存：

  $$
  a = T(\text{FLOAT32}(from)), \quad b = T(\text{FLOAT32}(to)), \quad d = T(b - a)
  $$

  其中$x_i$为Philox4x32-10算法生成的原始uint32随机数，$u_i$与候选值$v_i$的计算如下：

  $$
  u_i = \text{FLOAT32}\left(\text{FLOAT32}(x_i) \times 2^{-32} + 2^{-33}\right), \quad
  v_i = T\left(\text{FLOAT32}(u_i) \times \text{FLOAT32}(d) + \text{FLOAT32}(a)\right)
  $$

  候选值先转换回输出数据类型，再用`==`与上界$b$比较：相等时改写为下界$a$，否则保留$v_i$：

  $$
  y_i = \begin{cases}
  a, & v_i = b \\
  v_i, & v_i \neq b
  \end{cases}
  $$

  $u_i$的归一化方式与竞品curand_uniform一致，即$u = x \times 2^{-32} + 2^{-33}$（FLOAT32计算）；受FLOAT32舍入影响，$u_i$在FLOAT32下可能取到1，FLOAT16/BFLOAT16的候选值也可能在转换为输出类型后等于上界；当候选值恰好等于上界$b$时触发上述上界回绕。因此$[from, to)$应按转换后的边界解释：当$a < b$时通常$y_i \in [a, b)$，但受浮点舍入影响，$u_i$接近1时候选值可能略高于$b$且不等于$b$（不回绕），此时$y_i$可略超$b$，下界$a$始终成立；当$a = b$（$from$、$to$转换为输出类型后重合，退化情形）时，$y_i$恒为$a$。上述计算过程不改变Philox随机序列的生成规则，且仅适用于本算子及其支持产品。

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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状，1-D tensor。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>Philox算法的随机数种子，0-D标量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>Philox算法的偏移量，0-D标量。必须是4的倍数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>from</td>
      <td>输入</td>
      <td>均匀分布随机范围的下界（包含），0-D标量。</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>to</td>
      <td>输入</td>
      <td>均匀分布随机范围的上界（不包含），0-D标量。</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出指定形状的均匀分布随机值，值域为[from, to)，边界按转换为输出数据类型后的值解释（见计算公式）。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>输出数据类型，默认为FLOAT32。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- offset必须是4的倍数。
- 输出张量维度支持0~8维。
- from必须小于等于to，且to - from不能超出输出数据类型的表示范围。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_stateless_uniform](./examples/arch35/test_aclnn_stateless_uniform.cpp) | 通过[aclnnInplaceUniform](../dsa_random_uniform/docs/aclnnInplaceUniform.md)接口方式调用StatelessUniform算子。 |
| 图模式调用 | [test_geir_stateless_uniform](./examples/test_geir_stateless_uniform.cpp) | 通过[算子IR](./op_graph/stateless_uniform_proto.h)构图方式调用StatelessUniform算子。 |
