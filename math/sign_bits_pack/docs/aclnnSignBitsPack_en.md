# aclnnSignBitsPack

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:03:48.016Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Packs 1-bit Adam of the FLOAT16 or FLOAT32 data type into UINT8.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnSignBitsPackGetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call "aclnnSignBitsPack" to perform the computation.

```cpp
aclnnStatus aclnnSignBitsPackGetWorkspaceSize(
  const aclTensor  *self,
  int64_t           size,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSignBitsPack(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnSignBitsPackGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the floating-point tensor to be packed, from which the sign bit of each element is extracted.</td>
      <td><li>Empty tensors are supported.</li><li>NaN values are not supported.</li><li>Both +0 and -0 are treated as non-negative (sign bit is 0).</li></td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size (int64_t)</td>
      <td>Input</td>
      <td>Indicates the first dimension of the output tensor during reshape.</td>
      <td>The value must be a positive integer (size &ge; 1), and ceil(N/8) must be divisible by size.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the sign bit packing result.</td>
      <td><li>Empty tensor is supported.</li><li>The first dimension of the output must be equal to size.</li><li>When the number of elements in self is not divisible by 8, the total output length is (number of elements in self // 8) + 1; when the number of elements in self is divisible by 8, the total output length is (number of elements in self / 8).</li></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data format of self or out is not ND.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 1, or the dimension of out is not 2.</td>
    </tr>
    <tr>
      <td>size is less than or equal to 0, or ceil(N/8) is not divisible by size.</td>
    </tr>
    <tr>
      <td>The size of the first dimension of out is not equal to size.</td>
    </tr>
    </tbody></table>

## aclnnSignBitsPack

- **Parameters**

    <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
    </colgroup>
    <thead>
            <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the Device.</td></tr>
            <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnSignBitsPackGetWorkspaceSize.</td></tr>
            <tr><td>executor</td><td>Input</td><td>Executor of the op, containing the operator computation flow.</td></tr>
            <tr><td>stream</td><td>Input</td><td>Stream for executing tasks.</td></tr>
        </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSignBitsPack defaults to a deterministic implementation.
- Only the ND data format is supported.
- The input self must be 1D (rank=1), and the output out must be 2D (rank=2).
- The type combination is fixed to FLOAT16→UINT8 or FLOAT→UINT8. Other dtype combinations are not supported.
- The attribute size must be a positive integer (size &ge; 1), and ceil(N/8) must be divisible by size.
- The first dimension size of output out must be equal to size.
- Empty tensors (N=0) are supported. An empty output with shape [size, 0] is returned directly.
- Non-contiguous tensors are supported.
- Both +0 and -0 are treated as non-negative (the sign bit is 0). NaN is not supported, and the behavior is undefined.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sign_bits_pack.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t outsize = 2;
    std::vector<int64_t> selfShape = {16};
    std::vector<int64_t> outShape = {2, 1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                                       -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0};
    std::vector<uint8_t> outHostData = {0, 0};

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSignBitsPackGetWorkspaceSize(self, outsize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsPackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSignBitsPack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignBitsPack failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto outSize = GetShapeSize(outShape);
    std::vector<uint8_t> resultData(outSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint8_t), outDeviceAddr,
                      outSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outSize; i++) {
        LOG_PRINT("result[%ld] is: %u\n", i, (unsigned int)resultData[i]);
    }

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
