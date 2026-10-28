# aclnnAmpUpdateScale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:18:03.413Z -->

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

- Description: Implements the dynamic Scale update logic in AMP (Automatic Mixed Precision) training. Dynamically adjusts the loss scale size based on the current scale value, the growth tracker counter, and whether Inf/NaN is found.

- Formula:

    $$
    \text{updated\_scale} = \begin{cases}
    \text{current\_scale} \times \text{backoff\_factor} & \text{if found\_inf} \neq 0 \\
    \text{current\_scale} \times \text{growth\_factor} & \text{if growth\_tracker + 1 = growth\_interval and new\_scale is finite} \\
    \text{current\_scale} & \text{otherwise}
    \end{cases}
    $$

    $$
    \text{updated\_growth\_tracker} = \begin{cases}
    0 & \text{if found\_inf} \neq 0 \text{ or growth triggered} \\
    \text{growth\_tracker} + 1 & \text{otherwise}
    \end{cases}
    $$

    Description:

    - If found_inf is not 0, the scale is multiplied by backoff_factor to decrease, and growth_tracker is reset to 0
    - If found_inf is 0 and growth_tracker + 1 equals growth_interval, the scale is multiplied by growth_factor to increase
    - If the increased new_scale overflows (inf/nan), the current scale remains unchanged, and growth_tracker is reset to 0
    - In other cases, the scale remains unchanged, and growth_tracker increments by 1

- Use case: dynamic loss scaling (Dynamic Loss Scaling) in AMP training, to prevent gradient underflow in FP16/BF16 mixed-precision training.

## Function Prototype

Each operator is designed as a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnAmpUpdateScaleGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process. Then, call "aclnnAmpUpdateScale" to perform the computation.

```Cpp
aclnnStatus aclnnAmpUpdateScaleGetWorkspaceSize(
  const aclTensor* currentScale,
  const aclTensor* growthTracker,
  const aclTensor* foundInf,
  double           growthFactor,
  double           backoffFactor,
  int64_t          growthInterval,
  const aclTensor* updatedScale,
  const aclTensor* updatedGrowthTracker,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAmpUpdateScale(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnAmpUpdateScaleGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 160px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>currentScale (aclTensor*)</td>
      <td>Input</td>
      <td>Current loss scale value.</td>
      <td>The shape is scalar [1].</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>growthTracker (aclTensor*)</td>
      <td>Input</td>
      <td>Counter of consecutive steps without Inf/NaN.</td>
      <td>The shape is scalar [1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>foundInf (aclTensor*)</td>
      <td>Input</td>
      <td>Flag indicating whether Inf/NaN is detected.</td>
      <td>The shape is scalar [1]. 0 indicates normal, and non-0 indicates that Inf/NaN is found. The data type must be the same as that of currentScale.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>growthFactor (float)</td>
      <td>Input</td>
      <td>Scale growth factor.</td>
      <td>When no Inf/NaN is detected for consecutive growth_interval steps, scale is multiplied by this factor. Usually set to 2.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backoffFactor (float)</td>
      <td>Input</td>
      <td>Scale backoff factor.</td>
      <td>When Inf/NaN is detected, the scale is multiplied by this factor. Typically set to 0.5.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>growthInterval (int64_t)</td>
      <td>Input</td>
      <td>Interval steps for triggering scale growth.</td>
      <td>That is, scale increases after Inf/NaN is not detected for this number of consecutive steps. Value range: >= 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>updatedScale (aclTensor*)</td>
      <td>Output</td>
      <td>Updated loss scale value.</td>
      <td>Shape is [1]. The data type must be the same as that of currentScale.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>updatedGrowthTracker (aclTensor*)</td>
      <td>Output</td>
      <td>Updated growth tracker counter.</td>
      <td>Shape is [1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
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
      <td>Returns the op executor, including the operator computation flow.</td>
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>Shapes of currentScale, growthTracker, and foundInf are not scalar [1].</td>
    </tr>
    <tr>
      <td>growthInterval is out of the range [1, 2147483647].</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>currentScale, growthTracker, foundInf, updatedScale, or updatedGrowthTracker is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of currentScale is not supported.</td>
    </tr>
    <tr>
      <td>The data type of foundInf is inconsistent with currentScale.</td>
    </tr>
    <tr>
      <td>The data type of updatedScale is inconsistent with currentScale.</td>
    </tr>
    <tr>
      <td>The data type of growthTracker is not INT32.</td>
    </tr>
  </tbody>
  </table>

## aclnnAmpUpdateScale

- **Parameters:**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnAmpUpdateScaleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
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

## Constraints

- Deterministic computation:
  - aclnnAmpUpdateScale defaults to a deterministic implementation.

- **Data type constraint**: The data type of current_scale must be the same as that of found_inf; the data type of updated_scale must be the same as that of current_scale.
- **Shape constraint**: All input and output tensors are scalars, with shape [1].
- **growthInterval constraint**: The value range of growthInterval is [1, 2147483647].
- **Inf/NaN priority**: If found_inf is not 0, the fallback logic is executed directly, ignoring the status of growth_tracker.
- **Overflow protection**: If the new value of scale overflows (inf/nan), the current scale remains unchanged and growth_tracker is reset to 0.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_amp_update_scale.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Custom construction based on the API is required.
  std::vector<int64_t> scalarShape = {1};
  void* currentScaleDeviceAddr = nullptr;
  void* growthTrackerDeviceAddr = nullptr;
  void* foundInfDeviceAddr = nullptr;
  void* updatedScaleDeviceAddr = nullptr;
  void* updatedGrowthTrackerDeviceAddr = nullptr;
  aclTensor* currentScale = nullptr;
  aclTensor* growthTracker = nullptr;
  aclTensor* foundInf = nullptr;
  aclTensor* updatedScale = nullptr;
  aclTensor* updatedGrowthTracker = nullptr;

  // Create currentScale.
  std::vector<float> currentScaleHost = {65536.0f};
  ret = CreateAclTensor(currentScaleHost, scalarShape, &currentScaleDeviceAddr, ACL_FLOAT, &currentScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create growthTracker.
  std::vector<int32_t> growthTrackerHost = {900};
  ret = CreateAclTensor(growthTrackerHost, scalarShape, &growthTrackerDeviceAddr, ACL_INT32, &growthTracker);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create foundInf.
  std::vector<float> foundInfHost = {0.0f};
  ret = CreateAclTensor(foundInfHost, scalarShape, &foundInfDeviceAddr, ACL_FLOAT, &foundInf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output updatedScale.
  std::vector<float> updatedScaleHost = {0.0f};
  ret = CreateAclTensor(updatedScaleHost, scalarShape, &updatedScaleDeviceAddr, ACL_FLOAT, &updatedScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output updatedGrowthTracker.
  std::vector<int32_t> updatedGrowthTrackerHost = {0};
  ret = CreateAclTensor(updatedGrowthTrackerHost, scalarShape, &updatedGrowthTrackerDeviceAddr, ACL_INT32,
                        &updatedGrowthTracker);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the first-phase API to obtain the workspace size and executor.
  float growthFactor = 2.0f;
  float backoffFactor = 0.5f;
  int64_t growthInterval = 1000;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnAmpUpdateScaleGetWorkspaceSize(currentScale, growthTracker, foundInf, growthFactor, backoffFactor,
                                            growthInterval, updatedScale, updatedGrowthTracker, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 4. Allocate workspace memory based on workspaceSize.
  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 5. Call the second-phase API to execute the computation.
  ret = aclnnAmpUpdateScale(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAmpUpdateScale failed. ERROR: %d\n", ret); return ret);

  // 6. Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 7. Copy the output data from the device to the host and print the result.
  float updatedScaleVal = 0.0f;
  int32_t updatedGrowthTrackerVal = 0;
  ret = aclrtMemcpy(&updatedScaleVal, sizeof(float), updatedScaleDeviceAddr, sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedScale failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(&updatedGrowthTrackerVal, sizeof(int32_t), updatedGrowthTrackerDeviceAddr, sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy updatedGrowthTracker failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("aclnnAmpUpdateScale result: updatedScale = %f, updatedGrowthTracker = %d\n", updatedScaleVal,
            updatedGrowthTrackerVal);

  // 8. (boilerplate) Release resources.
  aclDestroyTensor(currentScale);
  aclDestroyTensor(growthTracker);
  aclDestroyTensor(foundInf);
  aclDestroyTensor(updatedScale);
  aclDestroyTensor(updatedGrowthTracker);
  aclrtFree(currentScaleDeviceAddr);
  aclrtFree(growthTrackerDeviceAddr);
  aclrtFree(foundInfDeviceAddr);
  aclrtFree(updatedScaleDeviceAddr);
  aclrtFree(updatedGrowthTrackerDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspace);
  }
  aclrtDestroyStream(stream);
  auto aclRet = aclrtResetDevice(deviceId);
  CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", aclRet); return aclRet);
  aclRet = aclFinalize();
  CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", aclRet); return aclRet);
  return 0;
}
```
