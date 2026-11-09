# aclRfft1D

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Performs the RFFT (Fourier transform) computation on the input tensor self. The output is a complex tensor that contains non-negative frequencies.

### Formula

$$
{\displaystyle X_{k}=\sum _{n=0}^{N-1}x_{n}\cdot e^{-i2\pi {\tfrac {k}{N}}n}}
$$

### Example

Assume that self is {1, 2, 3, 4}. Then out = {10, 0, -2, 2, -2, 0} = {10 + 0j, -2 + 2j, -2 + 0j} (customized)

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclRfft1DGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclRfft1D" API to execute the computation.

```Cpp
aclnnStatus aclRfft1DGetWorkspaceSize(
  const aclTensor*  self,
  int64_t           n,
  int64_t           dim,
  int64_t           norm,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclRfft1D(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclRfft1DGetWorkspaceSize

- **Parameters:**

    <table style="table-layout: fixed; width: 1200px"><colgroup>
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 350px">
    <col style="width: 80px">
    <col style="width: 80px">
    <col style="width: 80px">
    </colgroup>
      <thead>
        <tr>
          <th>Parameter (Type)</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Instruction</th>
          <th>Data Type</th>
          <th>Format</th>
          <th>shape</th>
          <th>Non-contiguous tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>self (aclTensor*)</td>
          <td>Input</td>
          <td>Input in the formula.</td>
          <td>-</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>1-7 dimensions</td>
          <td>Supported</td>
        </tr>
        <tr>
          <td>n (int64_t)</td>
          <td>Input</td>
          <td>Signal length.</td>
          <td>If specified, the input is padded with zeros or trimmed to this length before the Rfft1D computation. The value range of n is [1, 4096], and the maximum value of 2 to the power of n is 262144.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>dim (int64_t)</td>
          <td>Input</td>
          <td>Dimension.</td>
          <td>If specified, the RFFT is applied to the specified dimension. The supported values are [-self.dim(), self.dim()-1].</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>norm (int64_t)</td>
          <td>Input</td>
          <td>Normalization mode.</td>
          <td>Supported values: 1 indicates no normalization, 2 indicates normalization by 1/n, and 3 indicates normalization by 1/sqrt(n).</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>out (aclTensor*)</td>
          <td>Output</td>
          <td>Output in the formula.</td>
          <td>-</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>-</td>
          <td>Supported</td>
        </tr>
        <tr>
          <td>workspaceSize (uint64_t*)</td>
          <td>Output</td>
          <td>Returns the size of the workspace to be allocated on the Device side.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor (aclOpExecutor**)</td>
          <td>Output</td>
          <td>Returns the op executor, which contains the operator computation flow.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1150px">
    <colgroup>
    <col style="width: 380px">
    <col style="width: 135px">
    <col style="width: 635px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input or output tensor is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type and dimensions of self are not within the supported range.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>The intermediate result is null.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>The executor is null.</td>
      </tr>
    </tbody>
    </table>

## aclRfft1D

- **Parameters:**

    <table style="table-layout: fixed; width: 1200px"><colgroup>
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 350px">
    <col style="width: 80px">
    <col style="width: 80px">
    <col style="width: 80px">
    </colgroup>
      <thead>
        <tr>
          <th>Parameter (Type)</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Instruction</th>
          <th>Data Type</th>
          <th>Format</th>
          <th>shape</th>
          <th>Non-contiguous tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>workspace (void*)</td>
          <td>Input</td>
          <td>Memory address of the workspace allocated on the Device side.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>workspaceSize (uint64_t)</td>
          <td>Input</td>
          <td>Size of the workspace allocated on the Device side.</td>
          <td>Obtained by calling the first-phase API aclRfft1DGetWorkspaceSize.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor (aclOpExecutor*)</td>
          <td>Input</td>
          <td>op executor, which contains the operator computation flow.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>stream (aclrtStream)</td>
          <td>Input</td>
          <td>Stream for executing tasks.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclRfft1D defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compiling and running, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/acl_rfft1d.h"
#define CHECK_RET(cond, return_expr)   \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)
#define LOG_PRINT(message, ...)       \
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

    std::vector<int64_t> selfShape = {1, 1, 8};
    std::vector<int64_t> outShape = {1, 1, 5, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int n = 8;
    int dim = -1;
    int norm = 1; // backward - 1, forward - 2, ortho - 3

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclRfft1DGetWorkspaceSize(self, n, dim, norm, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1DGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclRfft1D(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclRfft1D failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
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
