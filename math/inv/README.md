# Inv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量的每个元素逐元素计算倒数。

- 计算公式：

$$
y_{i}=\frac{1}{x_{i}}
$$

- 浮点（FLOAT、FLOAT16、BFLOAT16）按IEEE-754真值倒数计算（除零得 $\pm\text{Inf}$、$\pm\text{Inf}$ 得 $\pm0$、NaN得NaN）。

- INT32按截断向零的整数倒数计算，整型下 $1/x$ 仅当 $|x|=1$ 时非零，等价为三值映射：

$$
y_{i}=
\begin{cases}
+1 & x_{i}=+1 \\
-1 & x_{i}=-1 \\
0  & \text{其他（含 } x_{i}=0、|x_{i}|\ge 2、\text{INT\_MIN}、\text{INT\_MAX)}
\end{cases}
$$

  其中 $x_{i}=0$ 映射为 $0$ 且不抛异常，$\text{INT\_MIN}(-2147483648)$、$\text{INT\_MAX}(2147483647)$ 均映射为 $0$ 且不溢出。

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
      <td>待进行inv计算的入参，公式中的x。支持空Tensor、非连续Tensor，shape维度0-8。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行inv计算的出参，公式中的y。shape和数据类型与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x与y必须shape、dtype相同，逐元素计算，无广播。
- 动态shape支持情况：支持动态shape与动态rank（-1未知维度 / -2未知rank），维度与rank在运行时由实际输入具体化，具体化后须满足shape维度0-8。
- 浮点路径精度按社区标准：FLOAT rtol/atol = 1e-4，FLOAT16 = 1e-3，BFLOAT16 = 4e-3。
- INT32路径为精确匹配（bitwise，rtol = atol = 0），采用纯整型比较 + 选择实现三值映射，不经float32中转、不对INT_MIN取负，故INT_MIN/INT_MAX不溢出、x = 0不抛异常。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|--------------|--------------|
| GE图模式 | [test_geir_inv](./examples/test_geir_inv.cpp) | 通过[算子IR](op_graph/inv_proto.h)构图方式调用Inv算子。 |
