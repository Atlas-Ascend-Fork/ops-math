# aclnnViewCopy

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2训练系列产品 | √ |

## 功能描述

`aclnnViewCopy` 提供 ViewCopy 算子的两段式 ACLNN L2 接口。该接口将 `src` 的逻辑视图拷贝到 `dst` 的逻辑视图位置，输出 Tensor 仍命名为 `dst`，shape 与输入 `dst` 一致。

## 头文件与构建来源

本页描述 `experimental/conversion/view_copy` 的实验接口，需要启用 `--experimental` 并选择 `--ops=view_copy` 构建。头文件为 [aclnn_view_copy.h](../op_api/aclnn_view_copy.h)，应与同次构建的算子库配套使用；自定义包安装后位于 `vendors/<vendor_name>_math/op_api/include/`，内置包位于 `<架构>-linux/include/aclnnop/`。构建和安装步骤参见[算子调用](../../../../docs/zh/install/compile.md)。

常规构建的同名接口由 `conversion/contiguous/op_host/op_api/aclnn_contiguous.cpp` 实现，其第一段接收 `const aclTensorList* in`、`const aclTensorList* out`、`uint64_t* workspaceSize`、`aclOpExecutor** executor` 四个参数，声明见[常规接口头文件](../../../../conversion/contiguous/op_host/op_api/aclnn_view_copy.h)。它与本页的多 Tensor 参数接口签名不同。动态库中存在同名符号不能证明其使用本页签名，请使用与实际构建来源匹配的头文件和库。

## 函数原型

```cpp
aclnnStatus aclnnViewCopyGetWorkspaceSize(
    const aclTensor *dst,
    const aclTensor *dst_size,
    const aclTensor *dst_stride,
    const aclTensor *dst_storage_offset,
    const aclTensor *src,
    const aclTensor *src_size,
    const aclTensor *src_stride,
    const aclTensor *src_storage_offset,
    const aclTensor *dst_out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnViewCopy(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnViewCopyGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | shape |
| :--- | :---: | :--- | :--- | :---: | :--- |
| dst | 输入 | 原始目标存储 Tensor | FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、BOOL、INT64 | ND | 1-8 维 |
| dst_size | 输入 | 目标视图逻辑 shape | INT32、INT64 | ND | 元素个数为 rank |
| dst_stride | 输入 | 目标视图 stride | INT32、INT64 | ND | 元素个数为 rank |
| dst_storage_offset | 输入 | 目标视图 storage offset | INT32、INT64 | ND | 元素个数为 1 |
| src | 输入 | 源存储 Tensor | 与 `dst` 一致 | ND | 1-8 维 |
| src_size | 输入 | 源视图逻辑 shape | INT32、INT64 | ND | 元素个数为 rank |
| src_stride | 输入 | 源视图 stride | INT32、INT64 | ND | 元素个数为 rank |
| src_storage_offset | 输入 | 源视图 storage offset | INT32、INT64 | ND | 元素个数为 1 |
| dst_out | 输出 | 拷贝后的目标存储 Tensor | 与 `dst` 一致 | ND | 与 `dst` 一致 |
| workspaceSize | 输出 | 返回 device workspace 大小 | - | - | - |
| executor | 输出 | 返回执行器，供第二段接口使用 | - | - | - |

## aclnnViewCopy 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| workspace | 输入 | device 侧 workspace 地址，由第一段接口申请大小后分配 |
| workspaceSize | 输入 | workspace 大小，由第一段接口返回 |
| executor | 输入 | 第一段接口返回的执行器 |
| stream | 输入 | 执行任务的 ACL stream |

## 返回值

| 返回值 | 说明 |
| :--- | :--- |
| ACLNN_SUCCESS | 执行成功 |
| ACLNN_ERR_PARAM_NULLPTR | 输入、输出、`workspaceSize` 或 `executor` 为空指针 |
| ACLNN_ERR_PARAM_INVALID | dtype、shape、rank、metadata 长度等参数不满足约束 |
| ACLNN_ERR_INNER | 内部执行或 metadata 读取失败 |

## 约束与限制

- `dst`、`src`、`dst_out` 的数据类型必须一致。
- metadata Tensor 数据类型仅支持 `INT32`、`INT64`，且六个 metadata Tensor 数据类型必须一致。
- `dst_size` 和 `src_size` 的元素个数表示逻辑视图 rank，rank 范围为 1 到 8。
- `dst_stride` 和 `src_stride` 的元素个数必须等于 rank。
- `dst_storage_offset` 和 `src_storage_offset` 的元素个数必须为 1。
- 当前仅支持 ND 格式，不支持广播。

## 调用示例

参考 [test_aclnn_view_copy.cpp](../examples/test_aclnn_view_copy.cpp)。
