# aclnnLerp&aclnnInplaceLerp

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:23:35.513Z -->

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

- Description: Performs linear interpolation between the start and end tensors based on a given weight, and returns the interpolated tensor.

- Formula:

$$
\text { out }_i=\text { start }_i+\text { weight }_i \times\left(\text { end }_i-\text { start }_i\right)
$$

## Function Prototype

- aclnnLerp and aclnnInplaceLerp implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnLerp: Create a new output tensor object to store the computation result.
  - aclnnInplaceLerp: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnLerpGetWorkspaceSize" or "aclnnInplaceLerpGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that contains the computation process. Then call "aclnnLerp" or "aclnnInplaceLerp" to perform the computation.

```Cpp
aclnnStatus aclnnLerpGetWorkspaceSize(
  const aclTensor*          self,
  const aclTensor*          end,
  const aclTensor*          weight,
  aclTensor*                out,
  uint64_t*                 workspaceSize,
  aclOpExecutor**           executor)
```

```Cpp
aclnnStatus aclnnLerp(
  void*                     workspace,
  uint64_t                  workspaceSize,
  aclOpExecutor*            executor,
  const aclrtStream         stream)
```

```Cpp
aclnnStatus aclnnInplaceLerpGetWorkspaceSize(
  aclTensor*                selfRef,
  const aclTensor*          end,
  const aclTensor*          weight,
  uint64_t*                 workspaceSize,
  aclOpExecutor**           executor)
```

```Cpp
aclnnStatus aclnnInplaceLerp(
  void*                     workspace,
  uint64_t                  workspaceSize,
  aclOpExecutor*            executor,
  const aclrtStream         stream)
```

## aclnnLerpGetWorkspaceSize

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
      <th>Dimension</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>The input start in the formula.</td>
      <td>The data type must be the same as that of end, weight, and out. The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with end and weight.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>end (aclTensor*)</td>
      <td>Input</td>
      <td>The input end in the formula.</td>
      <td>The data type must be the same as self, weight, and out. The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self and weight.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>The input weight in the formula.</td>
      <td>The data type must be the same as self, end, and out. The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self and end.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>output.</td>
      <td>out in the formula.</td>
      <td>The data type must be the same as that of self, end, and weight. The shape must be the same as the shape after self, end, and weight are broadcast.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, end, weight, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of self, end, weight, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self, end, weight, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>self, end, and weight cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of self, end, and weight after broadcast is inconsistent with the shape of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnLerp

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnLerpGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLerpGetWorkspaceSize

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
      <th>data type</th>
      <th>data format</th>
      <th>Dimensions</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef(aclTensor*)</td>
      <td>Input/Output</td>
      <td>The input start and output out in the formula.</td>
      <td>The data type is the same as end and weight. The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with end and weight, and the shape after broadcasting must be the same as selfRef.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>end (aclTensor*)</td>
      <td>Input</td>
      <td>The input end in the formula.</td>
      <td>The data type must be the same as selfRef and weight. The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef and weight.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>The weight input in the formula.</td>
      <td>The data type must be the same as selfRef and end. The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef and end.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id8 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

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
      <th>Return Values</th>
      <th>Error Codes</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>selfRef, end, and weight are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of selfRef, end, and weight are out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of selfRef, end, and weight are inconsistent.</td>
    </tr>
    <tr>
      <td>selfRef, end, and weight cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape after broadcasting selfRef, end, and weight is inconsistent with the shape of selfRef.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceLerp

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnInplaceLerpGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLerp&aclnnInplaceLerp default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

aclnnLerp

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lerp_tensor.h"

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
    // Boilerplate: resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
              aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Device/stream initialization. Refer to the acl API documentation.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> endShape = {4, 2};
    std::vector<int64_t> weightShape = {1};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* endDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* end = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> endHostData = {4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> weightHostData = {2};
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the end aclTensor.
    ret = CreateAclTensor(endHostData, endShape, &endDeviceAddr, aclDataType::ACL_FLOAT, &end);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnLerp.
    ret = aclnnLerpGetWorkspaceSize(self, end, weight, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLerpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnLerp.
    ret = aclnnLerp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLerp failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar.
    aclDestroyTensor(self);
    aclDestroyTensor(end);
    aclDestroyTensor(weight);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify based on the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(endDeviceAddr);
    aclrtFree(weightDeviceAddr);
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

aclnnInplaceLerp

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lerp_tensor.h"

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
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
              aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> endShape = {4, 2};
    std::vector<int64_t> weightShape = {1};
    void* selfDeviceAddr = nullptr;
    void* endDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* end = nullptr;
    aclTensor* weight = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> endHostData = {4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> weightHostData = {2};
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the end aclTensor.
    ret = CreateAclTensor(endHostData, endShape, &endDeviceAddr, aclDataType::ACL_FLOAT, &end);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API
    // Call the first-phase API of aclnnInplaceLerp
    ret = aclnnInplaceLerpGetWorkspaceSize(self, end, weight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLerpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnInplaceLerp
    ret = aclnnInplaceLerp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLerp failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host memory
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar
    aclDestroyTensor(self);
    aclDestroyTensor(end);
    aclDestroyTensor(weight);

    // 7. Release device resources, which needs to be modified based on the specific API definition
    aclrtFree(selfDeviceAddr);
    aclrtFree(endDeviceAddr);
    aclrtFree(weightDeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
