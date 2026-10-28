# aclnnSilentCheck

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:08:51.349Z -->

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
  The SilentCheck operator identifies whether a silent detection fault is triggered by comparing the input feature value (`val`) against absolute and relative thresholds. It also supports controlling whether to trigger an alarm or checkpoint-based training resumption upon a fault through the environment variable (`npuAsdDetect`) passed in from the framework side. By default (i.e., when `npuAsdDetect=1`), only logs are printed.
- Formula:

  - If the current input `val` is inf/nan, or `val` exceeds the absolute threshold `cThreshL1`, or the jump exceeds the relative threshold `cCoeffL1`, it is identified as an L1 fault. If the environment variable `npuAsdDetect` is 2, a log is printed and checkpoint-based training resumption is triggered. If the environment variable `npuAsdDetect` is 1, `sfdaRef` and `stepRef` are updated and the function returns normally.
  - If the current input `val` exceeds the absolute threshold `cThreshL2`, or the jump exceeds the relative threshold `cCoeffL2`, it is identified as an L2 fault. An alarm is printed, and `sfdaRef` and `stepRef` are updated before normal return.
  - If neither an L1 fault nor an L2 alarm is triggered, it is a normal case: if `npuAsdDetect` is 3, the eigenvalue is printed; otherwise, `sfdaRef` and `stepRef` are updated before normal return.
  - `sfdaRef` is [pre_val, min_val, max_val], representing [last detected val, historical minimum val, historical maximum val]. `stepRef` is the detection count, which increments by one for each detection.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSilentCheckGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnSilentCheck" to execute the computation.

```Cpp
aclnnStatus aclnnSilentCheckGetWorkspaceSize(
    const aclTensor *val,
    aclTensor *inputGradRef,
    aclTensor *sfdaRef,
    aclTensor *stepRef,
    const int32_t cMinSteps,
    const float cThreshL1,
    const float cCoeffL1,
    const float cThreshL2,
    const float cCoeffL2,
    const int32_t npuAsdDetect,
    aclTensor* result,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnSilentCheck(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnSilentCheckGetWorkspaceSize

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
      <th>Dimensions (shape)</th>
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
      <td>0</td>
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
      <td>sfdaRef</td>
      <td>Input</td>
      <td>Numerical value from the last check.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stepRef</td>
      <td>Input</td>
      <td>Current step number.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cMinSteps</td>
      <td>Input</td>
      <td>Minimum number of steps to trigger the jump detection.</td>
      <td>Currently, the recommended value is 7.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL1</td>
      <td>Input</td>
      <td>Absolute value that triggers the L1 fault threshold.</td>
      <td>The recommended value is 1000000.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cCoeffL1</td>
      <td>Input</td>
      <td>Jump value that triggers the L1 fault threshold.</td>
      <td>Current recommended value: 100000</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cThreshL2</td>
      <td>Input</td>
      <td>Absolute value that triggers the L2 fault threshold.</td>
      <td>Current recommended value: 10000 and cThreshL1 > cThreshL2</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cCoeffL2</td>
      <td>Input</td>
      <td>L2 fault threshold triggered by a value jump.</td>
      <td>Currently, the recommended value is 5000, and cCoeffL1>cCoeffL2.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>npuAsdDetect</td>
      <td>Input</td>
      <td>Environment variable.</td>
      <td>Available values: 1, 2, or 3</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>result</td>
      <td>Output</td>
      <td>Determines whether silent detection is triggered and the level of the triggered fault.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <td>val, inputGradRef, sfdaRef, or stepRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of val, inputGradRef, sfdaRef, or stepRef is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of val, inputGradRef, sfdaRef, or stepRef does not meet requirements.</td>
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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, which is obtained by the first-phase API aclnnSilentCheckGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSilentCheck defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silent_check.h"

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
    // Boilerplate, acl initialization
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

    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the list of external acl APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Perform custom construction based on the API.
    std::vector<int64_t> valShape = {};
    std::vector<int64_t> inputGradShape = {4, 2};
    std::vector<int64_t> sfdaShape = {3};
    std::vector<int64_t> stepShape = {1};
    std::vector<int64_t> resultShape = {1};
    void* valDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* sfdaDeviceAddr = nullptr;
    void* stepDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    aclTensor* val = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* sfda = nullptr;
    aclTensor* step = nullptr;
    aclTensor* result = nullptr;
    std::vector<float> valHostData = {160.0};
    std::vector<float> inputGradHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> sfdaHostData = {70.0, 200.0, 400.0};
    std::vector<int64_t> stepHostData = {7};
    std::vector<int32_t> resultHostData = {0};
    int32_t c_min_steps = 7;
    float c_thresh_l1 = 1000000;
    float c_coeff_l1 = 100000;
    float c_thresh_l2 = 10000;
    float c_coeff_l2 = 5000;
    int32_t npu_asd_detect = 3;

    // Create the val aclTensor.
    ret = CreateAclTensor(valHostData, valShape, &valDeviceAddr, aclDataType::ACL_FLOAT, &val);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the input_grad aclTensor.
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the sfda aclTensor.
    ret = CreateAclTensor(sfdaHostData, sfdaShape, &sfdaDeviceAddr, aclDataType::ACL_FLOAT, &sfda);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the step aclTensor.
    ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the result aclTensor.
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT32, &result);
    if (result == nullptr) {
        std::cout << "result is nullptr!" << std::endl;
        return 0;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific HostApi.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnSilentCheck.
    ret = aclnnSilentCheckGetWorkspaceSize(val, inputGrad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheckGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAdd.
    ret = aclnnSilentCheck(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilentCheck failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host side. Change the code based on the specific API definition.
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 6. Release aclTensor and aclScalar. Change the code based on the specific API definition.
    aclDestroyTensor(val);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(sfda);
    aclDestroyTensor(step);
    aclDestroyTensor(result);

    // 7. Release device resources. Modify as required by the specific API definition.
    aclrtFree(valDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(sfdaDeviceAddr);
    aclrtFree(stepDeviceAddr);
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
