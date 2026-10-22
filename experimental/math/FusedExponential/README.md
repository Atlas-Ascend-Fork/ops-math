# FusedExponential

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |

构建配置分别为 `ascend910b` / `ascend910_93`。

## 功能说明

将已经生成的均匀分布样本转换为指数分布样本，融合 neg、add、边界填充、log 和 mul。此算子不生成随机数，不修改随机数生成器状态。

```text
epsilon = finfo(u.dtype).eps / 2
out = -log(min(1 - u, 1 - epsilon)) / lambd
```

`epsilon` 在 FLOAT16、BFLOAT16、FLOAT32 下分别为 `2^-11`、`2^-8`、`2^-24`。

## 参数说明

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor* u, const aclScalar* lambd, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFusedExponential(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

| 参数 | 说明 |
| --- | --- |
| u | Device Tensor；FLOAT16、BFLOAT16 或 FLOAT32，ND，rank 0～8 |
| lambd | Host DOUBLE scalar，正率参数；`+inf` 对应全零输出 |
| out | 与 u 的 shape、dtype 相同；支持独立输出和原地输出 |
| workspaceSize / executor | 第一段接口返回的工作空间字节数及执行器 |
| stream | 第二段接口执行所在的 ACL stream |

完整接口说明见 [aclnnFusedExponential](docs/aclnnFusedExponential.md)。

## 约束说明

输入与输出均支持非连续布局。空张量不启动计算；rank 0 表示一个标量元素。读回输出之前应同步对应 stream。相同输入和参数得到逐位一致的结果。

输出支持可由 stride 排序证明元素不重叠的布局（包括连续、转置、带步长切片），拒绝内部重叠或无法证明不重叠的布局。连续输入输出可以完全原地，不支持部分地址重叠。非连续输入先转连续、非连续输出经临时结果写回，保证读写顺序。输入展开视图允许零 stride；输出不允许在非单元素维度使用零 stride。shape 必须为非负具体维度，字节数不得超出 INT64_MAX。

调用方保证 `u` 的采样域为 `[0, 1)`，ACLNN 层不扫描 Device 数据。按照任务书，`lambd <= 0` 或 NaN 的值校验属于 torch 适配层；调用 ACLNN 前应检查 `lambd > 0`。当输入的低精度表示舍入为 1 时，有限正率参数对应 `+inf` 输出，不对重尾端追加钳制。

空参数、非法 dtype、不匹配的 shape/dtype、超出支持范围的 rank 和非 ND 格式分别由接口参数检查处理。

## 实现

各 AI Vector Core 处理不重叠的连续区间，采用 UB 分块搬运。计算在 FLOAT32 中完成，最后一次收窄至输出 dtype。对 FLOAT32 输入，通过 `log(1-v)` 的舍入修正降低小值侧抵消误差，其中 `v=max(u, epsilon)`。FLOAT16/BFLOAT16 输入经该保护后，`1-v` 可由 FLOAT32 精确表示，因此省去恒等的补偿运算。Host 将 DOUBLE 率参数分解为尾数与二进制指数，避免直接把极端 DOUBLE 率参数转换为 FLOAT32 后溢出或下溢。

连续路径直接写入输出，不创建 GM 中间结果或 bool mask。非连续输入通过 Contiguous 转换，非连续输出通过 ViewCopy 写回；构建显式依赖仓内 ViewCopy，以保持其接口层与内核实现一致。

算法不使用 workspace。CANN 9.1.0 执行框架对零字节 workspace 描述保留固定的 512 字节保护及对齐空间，连续路径的第一段接口仍可能返回 512 字节；调用方必须按返回值分配，不可强制传入零字节。空张量路径返回 0。非连续布局的转换缓冲另计。

## 调用说明

| 调用方式 | 示例 | 说明 |
| --- | --- | --- |
| ACLNN | [test_aclnn_fused_exponential.cpp](examples/test_aclnn_fused_exponential.cpp) | 完整初始化、两段式调用、精度核对和资源释放；通过 DEVICE 选择逻辑设备 |

安装算子包并设置 `ASCEND_CUSTOM_OPP_PATH` 后，可在仓库根目录执行：

```bash
DEVICE=0 bash build.sh --run_example FusedExponential eager cust --vendor_name=fusedexp --experimental
```

### 构建与测试

Host/Tiling、shape/dtype 推导及 API 参数校验使用仓库 UT 框架，在仓库根目录执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh -u --ophost --opapi --soc=ascend910b --ops=FusedExponential --experimental -j8
```

UT 不替代设备精度和性能测试。设备构建与补充集成测试命令如下。

在仓库根目录执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --pkg --soc=ascend910b --vendor_name=fusedexp --ops=FusedExponential --experimental -j8
bash build_out/cann-ops-math-fusedexp_linux-aarch64.run --quiet --install-path=/path/to/opp
export ASCEND_CUSTOM_OPP_PATH=/path/to/opp/vendors/fusedexp_math
export ATK_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
DEVICE=0 python experimental/math/FusedExponential/tests/st/test_fused_exponential.py
```

补充测试覆盖边界率参数、rank、空张量、尾块、非连续输出、原地、确定性、非默认 stream 与异常元数据。任务验收另外使用官方自验包中的完整 1000 条精度、200 条性能和内存探针，补充测试不能替代官方矩阵。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| katiyaaa | 个人开发者 | FusedExponential | 2026/10 | 新增 Atlas A2/A3 实现、ACLNN 接口及测试 |
