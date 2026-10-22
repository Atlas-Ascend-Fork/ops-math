# aclnnFusedExponential

## 产品支持情况

支持 Atlas A2、Atlas A3，构建配置分别为 `ascend910b`、`ascend910_93`。

## 功能说明

将已有均匀分布样本转换为指数分布样本，不生成随机数，不改变随机数生成器状态。

```text
epsilon = finfo(u.dtype).eps / 2
out = -log(min(1 - u, 1 - epsilon)) / lambd
```

FLOAT16、BFLOAT16、FLOAT32 的 epsilon 分别为 `2^-11`、`2^-8`、`2^-24`。

## 函数原型

头文件：`aclnn_fused_exponential.h`。先调用第一段接口获取工作空间大小及执行器，再调用第二段接口提交计算。

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor* u, const aclScalar* lambd, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFusedExponential(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

## aclnnFusedExponentialGetWorkspaceSize

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| u | 输入 | Device Tensor，FLOAT16、BFLOAT16 或 FLOAT32，ND 格式，0～8 维，支持非连续布局 |
| lambd | 输入 | Host DOUBLE scalar，必须大于 0；允许正无穷，此时输出全零 |
| out | 输出 | Device Tensor，shape、dtype 与 u 相同，ND 格式；支持非连续布局和原地输出 |
| workspaceSize | 输出 | Device 工作空间的字节数，必须使用实际返回值分配 |
| executor | 输出 | 本次调用的执行器，由第二段接口使用 |

返回值：成功返回 `ACLNN_SUCCESS`；空参数返回 `ACLNN_ERR_PARAM_NULLPTR`；不支持的 dtype、格式、rank，shape/dtype 不匹配或检测到不安全的输出布局返回 `ACLNN_ERR_PARAM_INVALID`。执行器或内部张量创建失败返回相应内部错误，算子启动失败向调用方返回其状态码。

## aclnnFusedExponential

| 参数 | 方向 | 说明 |
| --- | --- | --- |
| workspace | 输入 | Device 工作空间地址；第一段返回大小为 0 时可传 nullptr |
| workspaceSize | 输入 | 第一段返回的工作空间大小 |
| executor | 输入 | 第一段返回的执行器 |
| stream | 输入 | 提交计算使用的 ACL stream |

返回值：返回 `CommonOpExecutorRun` 的执行状态；成功提交返回 `ACLNN_SUCCESS`。接口异步执行，读取输出前必须同步 stream，并检查同步结果。

## 约束说明

- 调用方保证 u 位于 `[0, 1)`；ACLNN 不扫描 Device 输入数据。低精度样本舍入为 1 时，有限正率参数对应正无穷输出，不额外钳制。
- 按任务书分层约定，torch 适配层负责检查 `lambd > 0`，拒绝零、负值及 NaN；直接调用 ACLNN 的调用方也必须满足这一前置条件。
- 支持空张量，空张量不启动计算且返回零工作空间；0 维张量表示一个标量。
- 输出支持可由 stride 排序证明元素不重叠的布局，包括连续、转置、带步长切片。内部重叠或无法证明不重叠的输出布局返回参数错误。连续输入输出支持完全原地，拒绝部分重叠；非连续路径先完成连续化/临时结果计算，再写回。shape 必须为非负具体维度，字节数不超过 INT64_MAX。
- 连续路径不使用随输入规模增长的 GM 中间结果。CANN 9.1.0 执行框架可能返回固定 512 字节保护及对齐空间，调用方仍须按返回值分配。非连续布局转换所需缓冲另计。
- 输入、输出及工作空间的 Device 存储必须在异步计算完成前保持有效。

## 调用说明

完整可执行示例见 [test_aclnn_fused_exponential.cpp](../examples/test_aclnn_fused_exponential.cpp)，包含资源创建、两段式调用、返回值检查、结果校验和资源释放。

构建、安装及仓库测试命令见 [README](../README.md#调用说明)。设备集成调用示例见 [test_fused_exponential.py](../tests/st/test_fused_exponential.py)。
