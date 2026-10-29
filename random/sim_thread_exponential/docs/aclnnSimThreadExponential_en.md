# aclnnSimThreadExponential

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:45:50.800Z -->

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

- Description: Generates random numbers following an exponential distribution with parameter lambda and fills them into the selfRef tensor.
- Formula:

  $$
  f(x) = -\frac{1}{\lambda} \ln(1 - u), u \sim \text{Uniform}(0, 1]
  $$

## Function Prototype

This operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSimThreadExponentialGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnSimThreadExponential" API to perform the computation.

```cpp
aclnnStatus aclnnSimThreadExponentialGetWorkspaceSize(
  aclTensor       *selfRef,
  int64_t          count,
  double           lambda,
  int64_t          seed,
  int64_t          offset,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSimThreadExponential(
  void              *workspace,
  uint64_t           workspace_size,
  aclOpExecutor     *executor,
  aclrtStream  stream)
```

## aclnnSimThreadExponentialGetWorkspaceSize

- **Parameters**

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1497px"><colgroup>
     <col style="width: 150px">
     <col style="width: 120px">
     <col style="width: 300px">
     <col style="width: 330px">
     <col style="width: 212px">
     <col style="width: 100px">
     <col style="width: 140px">
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
        <td>selfRef</td>
        <td>Input/Output</td>
        <td>f(x) in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>8 dimensions or less</td>
        <td>-</td>
      </tr>
      <tr>
        <td>count</td>
        <td>Input</td>
        <td>Number of random numbers to generate.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lambda</td>
        <td>Input</td>
        <td>lambda in the formula.</td>
        <td>Must be greater than 0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>seed</td>
        <td>Input</td>
        <td>Random number seed.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offset</td>
        <td>Input</td>
        <td>Random number offset.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to request on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
    </tbody></table>
    </div>
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Codes</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>selfRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="4"> 161002 </td>
      <td>The data type of selfRef is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of selfRef exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>lambda is less than or equal to 0.</td>
    </tr>

  </tbody></table>

## aclnnSimThreadExponential

- **Parameters**:

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 650px">
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
        <td>Workspace memory address requested on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied for on the Device side, obtained by the first-phase API aclnnSimThreadExponentialGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream used to execute the task.</td>
      </tr>
    </tbody></table>
    </div>

- **Return value**:
  - aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- lambda must be greater than 0.
- Deterministic computation:
  - aclnnSimThreadExponential defaults to a deterministic implementation.

<!-- npu="950" id7 -->
- Ascend 950PR&950DT products:
  - offset must be a multiple of 4.
  - count must be equal to the total number of elements in the selfRef tensor.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_sim_thread_exponential.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device memory.
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
    // 1. Boilerplate: device/stream initialization. Refer to the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle checks as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> selfRefShape = {2, 5};
    void* selfRefDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // Create a selfRef aclTensor.
    ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int64_t count = 10;
    double lambda = 1.0;
    int64_t seed = 5;
    int64_t offset = 0;
    // 3. Call the CANN operator library API. Replace it with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnSimThreadExponential first-phase API.
    ret = aclnnSimThreadExponentialGetWorkspaceSize(selfRef, count, lambda, seed, offset, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSimThreadExponentialGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the aclnnSimThreadExponential second-phase API.
    ret = aclnnSimThreadExponential(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSimThreadExponential failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
    auto size = GetShapeSize(selfRefShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(selfRef);

    // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(selfRefDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
