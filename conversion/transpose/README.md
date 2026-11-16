# Transpose

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT系列产品</term>          |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>             |    √     |
| <term>Atlas推理系列产品</term>                       |    √     |
| <term>Atlas训练系列产品</term>                       |    √     |
| <term>Kirin X90处理器系列产品</term>                       |    √     |
| <term>Kirin 9030处理器系列产品</term>                       |    √     |

## 功能说明

- 算子功能：对tensor的任意维度进行调换。如输入x是shape为[2, 3, 5]的tensor，perm为(2, 0, 1)，则输出y是shape为[5, 2, 3]的tensor。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述                                                         | 数据类型 | 数据格式 |
| :----- | :------------- | :----------------------------------------------------------- | :------- | :------- |
| x   | 输入张量       | 需要进行维度置换的输入张量。                                 | INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT4、HIFLOAT4_SCALE | ND       |
| perm   | 输入张量       | 表示x的维度的排列。元素个数必须等于x的维度数量，取值需在[-x的维度数量，x的维度数量-1]范围内，且各元素不能重复。 | INT32、INT64    | -        |
| y    | 输出           | 维度不超过8维，shape由perm和x的shape共同决定（y.shape[i] = x.shape[perm[i]]），dtype需要与x一致。 | 同x   | ND       |

- <term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：数据类型不支持FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT4、HIFLOAT4_SCALE。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：数据类型不支持INT16、UINT16、UINT32、INT64、UINT64、BFLOAT16、BOOL、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT4、HIFLOAT4_SCALE。

## 约束说明

- x的维度数量不超过8维。
- perm的元素个数必须等于x的维度数量，取值需在[-x的维度数量，x的维度数量-1]范围内，且各元素不能重复。
- y的shape由perm和x的shape共同决定，dtype必须与x一致。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_channel_shuffle](examples/test_aclnn_channel_shuffle.cpp) | 通过[aclnnChannelShuffle](docs/aclnnChannelShuffle.md)接口方式调用Transpose算子。 |
| aclnn接口 | [test_aclnn_permute](examples/test_aclnn_permute.cpp) | 通过[aclnnPermute](docs/aclnnPermute.md)接口方式调用Transpose算子。 |
| 图模式调用 | [test_geir_permute](examples/test_geir_permute.cpp) | 通过[算子IR](op_graph/permute_proto.h)构图方式调用Transpose算子。 |
| 图模式调用 | [test_geir_transpose](examples/test_geir_transpose.cpp) | 通过[算子IR](op_graph/transpose_proto.h)构图方式调用Transpose算子。 |
