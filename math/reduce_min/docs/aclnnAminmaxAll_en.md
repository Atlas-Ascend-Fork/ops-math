# aclnnAminmaxAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:42:28.699Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Returns the minimum and maximum values of the input tensor across all dimensions.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnAminmaxAllGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size, and then call the "aclnnAminmaxAll" API to perform the computation.

```Cpp
aclnnStatus aclnnAminmaxAllGetWorkspaceSize(
const aclTensor *self,
aclTensor       *minOut,
aclTensor       *maxOut,
uint64_t        *workspaceSize,
aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAminmaxAll(
void          *workspace,
uint64_t       workspaceSize,
aclOpExecutor *executor,
aclrtStream    stream)
```

## aclnnAminmaxAllGetWorkspaceSize

- **Parameters:**

    <table style="table-layout: fixed; width: 1446px"><colgroup>
    <col style="width: 154px">
    <col style="width: 121px">
    <col style="width: 261px">
    <col style="width: 188px">
    <col style="width: 323px">
    <col style="width: 120px">
    <col style="width: 133px">
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
        <td>self</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>-</td>
        <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>minOut</td>
        <td>Output</td>
        <td>Output min value tensor.</td>
        <td>Data type is the same as self.</td>
        <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxOut</td>
        <td>Output</td>
        <td>Maximum value tensor in the output.</td>
        <td>Data type is the same as self.</td>
        <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
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

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 288px">
    <col style="width: 114px">
    <col style="width: 747px">
    </colgroup>
    <thead>
      <tr>
        <th>Error Code</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>self, minOut, or maxOut is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of self is not in the supported range.</td>
      </tr>
      <tr>
        <td>The data type of minOut or maxOut is inconsistent with that of self.</td>
      </tr>
      <tr>
        <td>The shape of self, minOut, or maxOut has more than 8 dimensions.</td>
      </tr>
    </tbody>
    </table>

## aclnnAminmaxAll

- **Parameters:**

    <table style="table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
    <col style="width: 872px">
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
        <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by the first-phase API aclnnAminmaxAllGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, containing the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAminmaxAll defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_aminmax_all.h"

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
  // Boilerplate. Initialize resources.
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

  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> selfShape = {2, 3, 2};
  std::vector<int64_t> outShape = {1};
  void* selfDeviceAddr = nullptr;
  void* minOutDeviceAddr = nullptr;
  void* maxOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* minOut = nullptr;
  aclTensor* maxOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> minOutHostData = {0};
  std::vector<float> maxOutHostData = {0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(minOutHostData, outShape, &minOutDeviceAddr, aclDataType::ACL_FLOAT, &minOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maxOutHostData, outShape, &maxOutDeviceAddr, aclDataType::ACL_FLOAT, &maxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAminmaxAll.
  ret = aclnnAminmaxAllGetWorkspaceSize(self, minOut, maxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAminmaxAllGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAminmaxAll.
  ret = aclnnAminmaxAll(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAminmaxAll failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the results from the device to the host. Change it based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> minResultData(size, 0);
  ret = aclrtMemcpy(minResultData.data(), minResultData.size() * sizeof(minResultData[0]), minOutDeviceAddr,
                    size * sizeof(minResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, minResultData[i]);
  }
  std::vector<float> maxResultData(size, 0);
  ret = aclrtMemcpy(maxResultData.data(), maxResultData.size() * sizeof(maxResultData[0]), maxOutDeviceAddr,
                    size * sizeof(maxResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, maxResultData[i]);
  }

  // 6. Release aclTensor. Change it based on the definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(minOut);
  aclDestroyTensor(maxOut);

  // 7. Release device resources. Change it based on the definition of the specific API.
  aclrtFree(selfDeviceAddr);
  aclrtFree(minOutDeviceAddr);
  aclrtFree(maxOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
