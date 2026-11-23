# Flatten

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    √     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |

## 功能说明

- **算子功能**：将输入Tensor基于给定的axis扁平化为一个2D的Tensor。
- **计算公式**：
  - 若输入x的shape为(d₀, d₁, ..., dₙ)，则输出y的shape为(d₀×d₁×...×dₐₓᵢₛ₋₁, dₐₓᵢₛ×dₐₓᵢₛ₊₁×...×dₙ)
  - 若axis取值为0，则输出y的shape为(1, d₀×d₁×...×dₙ)
  - 若axis取值为rank(x)，则输出y的shape为(d₀×d₁×...×dₙ, 1)

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 输入张量。 | ALL | ND |
| axis | 可选属性 | 扁平化基准轴，取值范围为[-rank(x), rank(x)]，默认值为1。 | INT | - |
| y | 输出 | 扁平化后的2D张量，数据类型与 `x` 一致。 | 与 `x` 相同 | ND |

## 约束说明

- `axis` 必须在 `[-rank(x), rank(x)]` 范围内（两端均包含），超出范围时推导失败。
- 输入为未知rank（-2）时，输出shape同样为-2。
- 参与乘积的维度中存在未知维度（-1）时，对应输出维度为-1。
- 输入为1维且该维为未知维度（-1）时，输出保持1维（-1）。
- aclnn接口当前仅支持axis = 1，输入维度范围为2D~8D。
- 支持的数据类型以 [`flatten_proto.h`](op_graph/flatten_proto.h) 中的IR定义为准。

## 调用说明

| 调用方式  | 样例代码                                                | 说明                                                         |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| 图模式调用 | [test_geir_flatten](./examples/test_geir_flatten.cpp) | 通过[算子IR](./op_graph/flatten_proto.h)构图方式调用Flatten算子。 |
| aclnn接口 | [test_aclnn_flatten](./examples/test_aclnn_flatten.cpp) | 通过[aclnnFlatten](docs/aclnnFlatten.md)接口方式调用Flatten算子。 |
