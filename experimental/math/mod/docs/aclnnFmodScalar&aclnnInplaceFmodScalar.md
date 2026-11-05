# aclnnFmodScalar & aclnnInplaceFmodScalar

## 功能说明

对 `self` 和 host scalar `other` 执行 Mod 取余计算：`out = self - other * trunc(self / other)`。`out` shape 需要与 `self` 一致。

## 接口原型

```cpp
aclnnStatus aclnnFmodScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnInplaceFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## 约束

- `self`、`out`支持 ND，维度不超过8。
- aclnn层支持DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、INT16类型推导；AI Core kernel覆盖BFLOAT16、FLOAT16、FLOAT32、INT32，其余类型走AI CPU fallback。BFLOAT16仅在支持该数据类型的NPU平台上走AI Core。
- Atlas A2系列产品、Atlas A3系列产品：支持INT16同数据类型计算、以及`self`/`other`分别为INT16与BFLOAT16/FLOAT16/FLOAT32（数据类型不同）的混合数据类型计算（由AI Core支持）。
- 已有的BFLOAT16/FLOAT16/FLOAT32/INT32同数据类型计算，以及DOUBLE/INT64/INT8/UINT8的AI CPU回退，在原支持产品上的行为保持不变。
- 精度：针对`self`/`other`商值较大（大\|self/other\|）的场景，Atlas A2系列产品/Atlas A3系列产品上的AI Core计算路径引入了数值稳定性增强算法，相比朴素截断取余（trunc-mod）实现降低了大商场景下的精度损失风险。
- `out`shape必须等于`self`shape。

## 样例

见 `../examples/test_aclnn_fmod_scalar.cpp`。
