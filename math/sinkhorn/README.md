# Sinkhorn

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：计算Sinkhorn最优传输计划，可以用于MoE模型中的专家路由。

- 计算公式：

  $$
  p=Sinkhorn(cost, tol)
  $$

- 初始化：

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1_old = d1 \\
  $$

- 重复执行：

  $$
  d0 = \frac{1}{R \times (sum(d1 \cdot cost, 1) + eps)} \\
  d1 = \frac{1}{C \cdot \sum(d0.unsqueeze(1) \cdot cost, 0) + eps)} \\
  error = mean(|d1_old - d1|) \\
  d1_old = d1
  $$

  直至满足：

  $$
  error \le tol
  $$

- 输出：

  $$
  p = d1 \cdot cost \cdot d0.unsqueeze(1)
  $$

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|cost|输入|表示成本张量，公式中的`cost`。二维矩阵，行数无固定上限，列数不超过4168，元素总数不超过UINT32_MAX。取值建议在[0,1]之间。|FLOAT、FLOAT16、BFLOAT16|ND|
|p|输出|表示最优传输张量，公式中的`p`。shape与cost一致。|FLOAT、FLOAT16、BFLOAT16|ND|
|tol|可选属性|表示计算Sinkhorn的误差，默认为0.0001。|FLOAT|-|

- <term>Atlas A2系列产品</term>：FP16输入值较大时可能较早产生Inf或NaN。

- <term>Ascend 950PR&950DT系列产品</term>：FP16输入的数值范围与FP32一致。

## 约束说明

- cost必须为二维矩阵且维度非负，列数不超过4168，行数无固定上限，元素总数不超过UINT32_MAX。
- cost取值建议在[0,1]之间，但该区间不是拒绝边界。域外值仍参与计算，可能在指数、行和或输出阶段产生Inf或NaN。
- tol支持全部FLOAT取值，NaN、±Inf和负值按浮点比较语义参与终止判断。
- 空Tensor合法。
- 迭代最大次数为10000，达到上限后返回第10000轮状态，不保证收敛。
- 终止判断严格使用`!(error > tol)`：NaN比较会终止，`+Inf`与有限tol比较时继续迭代，最终仍受10000次上限保护。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---------|------|------|
| aclnn API | [test_aclnn_sinkhorn](examples/test_aclnn_sinkhorn.cpp) | 通过[aclnnSinkhorn](docs/aclnnSinkhorn.md)接口方式调用Sinkhorn算子。|
| GEIR图模式 | - | Sinkhorn支持GEIR图模式。|
