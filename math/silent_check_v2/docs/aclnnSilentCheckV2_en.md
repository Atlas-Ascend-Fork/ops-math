# aclnnSilentCheckV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:08:26.350Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Operator function:
  aclnnSilentCheckV2 is newly added. The computation logic in this version differs from that of the [aclnnSilentCheck](../../silent_check/docs/aclnnSilentCheck_en.md) API. Based on the stepRef parameter, val is compared with and verified against the newly added Markov inequality threshold.
- Formula:

  - If stepRef==0:
    - If the current input `val` is inf/-inf/nan, or `val` exceeds `avgRef` * `cThreshL1`, it is identified as an L1 Level Fault, and an ERROR log is printed. If the environment variable `npuAsdDetect` is 1, `avgRef` and `stepRef` are updated and the function returns normally; otherwise, inputGradRef is set to zero and checkpoint-based training recovery is triggered.
    - If the current input `val` exceeds `avgRef` * `cThreshL2`, it is identified as an L2 Level Alarm; a WARNING log is printed, and `avgRef` and `stepRef` are updated before returning normally.
  - If stepRef > 0:
    - If the current input `val` is inf/-inf/nan, or `val` exceeds the Markov inequality threshold (avgRef/(1-beta1)^stepRef) * cThreshL1, it is identified as an L1 Level Fault, and an ERROR log is printed. If the environment variable `npuAsdDetect` is 2, checkpoint-based training recovery is triggered; if the environment variable `npuAsdDetect` is 1, `avgRef` and `stepRef` are updated and the function returns normally.
    - If the current input `val` exceeds the Markov inequality threshold (avgRef/(1-beta1)^stepRef) * cThreshL2, it is identified as an L2 Level Alarm; a WARNING log is printed, and `avgRef` and `stepRef` are updated before returning normally.
    - If no L1 Level Fault or L2 Level Alarm is triggered, under normal conditions: if `npuAsdDetect` is 3, the characteristic values of `val` are printed; otherwise, `avgRef` and `stepRef` are updated and the function returns normally.
  - `stepRef` indicates the detection count.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSilentCheckV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator. Then call the "aclnnSilentCheckV2" API to perform the computation.

```Cpp

aclnnStatus aclnnSilentCheckV2GetWorkspaceSize(
    const aclTensor *val,
    const aclTensor *max,
    aclTensor *avgRef,
    aclTensor *inputGradRef,
    aclTensor *stepRef,
    aclIntArray *dstSize,
    aclIntArray *dstStride,
    aclIntArray *dstOffset,
    float cThreshL1,
    float cThreshL2,
    float beta1,
    int32_t npuAsdDetect,
    aclTensor* result,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnSilentCheckV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnSilentCheckV2GetWorkspaceSize

- **Parameters**:
  <table style="table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
  <col style="width: 145px">
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
      <td>val</td>
      <td>Input</td>
      <td>Current input value.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>avgRef</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputGradRef</td>
      <td>Input</td>
      <td>Gradient tensor of the model input.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stepRef</td>
      <td>Input</td>
      <td>Current step.</td>
      <td>Negative values are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>Must be [1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstSize</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstStride</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstOffset</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL1</td>
      <td>Input</td>
      <td>Absolute value threshold that triggers an L1 fault.</td>
      <td>Current recommended value: 1000000.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL2</td>
      <td>Input</td>
      <td>Threshold of the absolute value that triggers an L2 fault.</td>
      <td>Currently, the recommended value is 10000 and cThreshL1>cThreshL2.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Input</td>
      <td>-</td>
      <td>The current recommended value is 0.99, and the range is 0&ltbeta1&lt1. The value is recommended to be as close to 1 as possible.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>npuAsdDetect</td>
      <td>Input</td>
      <td>Environment variable.</td>
      <td>-</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>result</td>
      <td>Output</td>
      <td>Returns the silent check result.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The input val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, and dstOffset are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of the input val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, and dstOffset are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of the input val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, and dstOffset do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The ranges of the input cThreshL1, cThreshL2, and beta1 do not meet the requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnSilentCheck

- **Parameters**:

  <table><thead>
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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnSilentCheckV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnSilentCheckV2 defaults to a deterministic implementation.

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silent_check_v2.h"

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

int64_t GetShapeSize(std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
  }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate: acl initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(std::vector<T>& hostData, std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to the device memory.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the list of acl APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on the API.
    std::vector<int64_t> valShape = {1};
    std::vector<int64_t> maxShape = {1};
    std::vector<int64_t> avgRefShape = {1};
    std::vector<int64_t> inputGradRefShape = {2, 3};
    std::vector<int64_t> stepRefShape = {1};
    std::vector<int64_t> resultShape = {1};
    void* valDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* avgRefDeviceAddr = nullptr;
    void* inputGradRefDeviceAddr = nullptr;
    void* stepRefDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    aclTensor* val = nullptr;
    aclTensor* max = nullptr;
    aclTensor* avgRef = nullptr;
    aclTensor* inputGradRef = nullptr;
    aclTensor* stepRef = nullptr;
    aclTensor* result = nullptr;
    std::vector<float> valHostData = {160.0};
    std::vector<float> maxHostData = {400.0};
    std::vector<float> avgRefHostData = {200.0};
    std::vector<float> inputGradRefHostData = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> stepRefHostData = {0};
    std::vector<int64_t> dstSizeData = {2, 3};
    std::vector<int64_t> dstStrideData = {3, 1};
    std::vector<int64_t> dstOffsetData = {0};
    std::vector<int32_t> resultHostData = {0};
    aclIntArray* dstSize = aclCreateIntArray(dstSizeData.data(), 2);
    aclIntArray* dstStride = aclCreateIntArray(dstStrideData.data(), 2);
    aclIntArray* dstOffset = aclCreateIntArray(dstOffsetData.data(), 1);
    float cThreshL1 = 1000000;
    float cThreshL2 = 10000;
    float beta1 = 0.99;
    int32_t npuAsdDetect = 3;

    // Create the val aclTensor.
    ret = CreateAclTensor(valHostData, valShape, &valDeviceAddr, aclDataType::ACL_FLOAT, &val);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the max aclTensor.
    ret = CreateAclTensor(maxHostData, maxShape, &maxDeviceAddr, aclDataType::ACL_FLOAT, &max);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the avgRef aclTensor.
    ret = CreateAclTensor(avgRefHostData, avgRefShape, &avgRefDeviceAddr, aclDataType::ACL_FLOAT, &avgRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the inputGradRef aclTensor.
    ret = CreateAclTensor(inputGradRefHostData, inputGradRefShape, &inputGradRefDeviceAddr, aclDataType::ACL_FLOAT, &inputGradRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the stepRef aclTensor.
    ret = CreateAclTensor(stepRefHostData, stepRefShape, &stepRefDeviceAddr, aclDataType::ACL_INT64, &stepRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the result aclTensor.
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT32, &result);
    if (result == nullptr) {
        std::cout << "result is nullptr!" << std::endl;
        return 0;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify it to the specific HostApi.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSilentCheckV2.
    ret = aclnnSilentCheckV2GetWorkspaceSize(val, max, avgRef, inputGradRef, stepRef, dstSize, dstStride, dstOffset, cThreshL1, cThreshL2, beta1, npuAsdDetect, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnSilentCheckV2.
    ret = aclnnSilentCheckV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from device memory to host memory. Modify it based on the specific API definition.
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
    aclDestroyTensor(val);
    aclDestroyTensor(max);
    aclDestroyTensor(avgRef);
    aclDestroyTensor(inputGradRef);
    aclDestroyTensor(stepRef);
    aclDestroyIntArray(dstSize);
    aclDestroyIntArray(dstStride);
    aclDestroyIntArray(dstOffset);
    aclDestroyTensor(result);

    // 7. Release device resources. Modify them based on the specific API definition.
    aclrtFree(valDeviceAddr);
    aclrtFree(maxDeviceAddr);
    aclrtFree(avgRefDeviceAddr);
    aclrtFree(inputGradRefDeviceAddr);
    aclrtFree(stepRefDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
