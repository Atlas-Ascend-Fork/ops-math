# aclnnBincount

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:28:36.520Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description:

  Computes the frequency of each value in a non-negative integer array. minlength specifies the minimum size of the output tensor. When weights is a null pointer, each occurrence of self[i] in self increments its frequency by 1. When weights is not null, each occurrence of self[i] increments its frequency by weights[i], and the result is stored at position self[i]+1 in out. Therefore, the size of out is the maximum of (the maximum value in self + 1) and minlength.

- Formula:

  If n is the value of self at position i, and weights is specified, then

  $$
  out[self_i] = out[self_i] + weights_i
  $$

  Otherwise:

  $$
  out[self_i] = out[self_i] + 1
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnBincountGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnBincount" to execute the computation.

```Cpp
aclnnStatus aclnnBincountGetWorkspaceSize(
  const aclTensor*        self,
  const aclTensor*        weights,
  int64_t                 minlength,
  aclTensor*              out,
  uint64_t*               workspaceSize,
  aclOpExecutor**         executor)
```

```Cpp
aclnnStatus aclnnBincount(
  void*                   workspace,
  uint64_t                workspaceSize,
  aclOpExecutor*          executor,
  const aclrtStream       stream)
```

## aclnnBincountGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>Must be non-negative integers.</td>
      <td>INT8, INT16, INT32, INT64, UINT8</td>
      <td>1-dim ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weights (aclTensor*)</td>
      <td>Input</td>
      <td>Weight of each value in self. Can be a null pointer.</td>
      <td>shape must be consistent with self.</td>
      <td>FLOAT, FLOAT16, FLOAT64, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
      <td>1D ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>minlength (int64_t)</td>
      <td>Input</td>
      <td>Specifies the minimum length of the output tensor.</td>
      <td>If the maximum value of self is less than minlength, the length of out is minlength. Otherwise, the length of out is the maximum value of self plus 1.</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The length of out is the maximum of the maximum value in self plus 1 and minlength.</td>
      <td>INT32, INT64, FLOAT, DOUBLE</td>
      <td>1D ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and data formats of self, out, and weights are not in the supported range.</td>
    </tr>
    <tr>
      <td>When weights is not null, the shapes of self and weights are inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnBincount

- **Parameters:**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device. The value is obtained by the first-phase API aclnnBincountGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b,910,310p" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: defaults to a deterministic implementation.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: defaults to a non-deterministic implementation. You can enable deterministic computation by calling aclrtSetSysParamOpt.
  <!-- end id8 -->

## Example

The following sample code is for reference only. For details about compiling and running, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bincount.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate, resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // Initialize device/stream. For details, see the acl API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> selfShape = {8};
    std::vector<int64_t> maxOutShape = {1};

    void* selfDeviceAddr = nullptr;
    aclTensor* self = nullptr;

    std::vector<int32_t> selfHostData = {8, 1, 2, 3, 4, 5, 6, 7};

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call bincount. outSize is fixed to 9 in this example.
    int64_t minlength = 0;
    int64_t outSize = 9;
    std::vector<int64_t> weightsShape = {8};
    std::vector<int64_t> outShape = {outSize};

    void* weightsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* weights = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> weightsHostData = {1, 1, 1.1, 2, 2, 2, 3, 3};
    std::vector<float> outHostData(outSize, 0);
    // Create the weights aclTensor.
    ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT, &weights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnBincount.
    ret = aclnnBincountGetWorkspaceSize(self, weights, minlength, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincountGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBincount.
    ret = aclnnBincount(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBincount failed. ERROR: %d\n", ret); return ret);

    // Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value, and copy the result from device memory to host memory.
    auto size = GetShapeSize(outShape);
    std::vector<float> bincountResultData(size, 0);
    ret = aclrtMemcpy(
        bincountResultData.data(), bincountResultData.size() * sizeof(bincountResultData[0]), outDeviceAddr,
        size * sizeof(bincountResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, bincountResultData[i]);
    }
    // Release aclTensor.
    aclDestroyTensor(self);
    aclDestroyTensor(weights);
    aclDestroyTensor(out);

    // Release resources.
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);

    aclrtFree(weightsDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
