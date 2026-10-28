# aclnnRoundDecimals&aclnnInplaceRoundDecimals

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:58:07.143Z -->

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

Rounds the elements of the input tensor to the specified number of decimal places.

## Function Prototype

- aclnnRoundDecimals and aclnnInplaceRoundDecimals implement the same function. The differences are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnRoundDecimals: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceRoundDecimals: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnRoundDecimalsGetWorkspaceSize" or "aclnnInplaceRoundDecimalsGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnRoundDecimals" or "aclnnInplaceRoundDecimals" to perform the computation.

  ```Cpp
  aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(
    const aclTensor* self,
    int64_t          decimals,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnRoundDecimals(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    aclrtStream       stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(
    aclTensor*       selfRef,
    int64_t          decimals,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceRoundDecimals(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    aclrtStream       stream)
  ```

## aclnnRoundDecimalsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 370px">
  <col style="width: 320px">
  <col style="width: 120px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>decimals (int64_t)</td>
      <td>Input</td>
      <td>Specifies the number of decimal places to round to.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Supports empty tensors. The shape must be the same as that of self.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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
  </tbody></table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**:

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
      <td>self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self or out is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnRoundDecimals

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 849px">
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
      <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnRoundDecimalsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnRoundDecimals&aclnnInplaceRoundDecimals default to a deterministic implementation.

- For scenarios where decimals is not 0:
  If the input data exceeds the range of (-347000, 347000), precision may be affected.

## aclnnInplaceRoundDecimalsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 370px">
  <col style="width: 120px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input tensor.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>decimals (int64_t)</td>
      <td>Input</td>
      <td>Specifies the number of decimal places to round to.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op Executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**:

  aclnnStatus: Return Status. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input selfRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of selfRef is out of the supported range.</td>
    </tr>
    <tr>
      <td>The number of dimensions of selfRef is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceRoundDecimals

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 849px">
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
      <td>Size of the workspace memory allocated on the Device side, obtained by the first-phase API aclnnInplaceRoundDecimalsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

When decimals is not 0: if the input data exceeds the range of (-347000, 347000), precision may be affected.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_round.h"

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
  // Boilerplate code for resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host side to the device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int decimals = 0;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  LOG_PRINT("Example of aclnnRoundDecimals API call \n");
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRoundDecimals.
  ret = aclnnRoundDecimalsGetWorkspaceSize(self, decimals, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoundDecimalsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRoundDecimals.
  ret = aclnnRoundDecimals(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoundDecimals failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device-side memory to host-side memory, and modify based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  LOG_PRINT("Example of aclnnInplaceRoundDecimals API call \n");
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceRoundDecimals.
  ret = aclnnInplaceRoundDecimalsGetWorkspaceSize(self, decimals, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRoundDecimalsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceRoundDecimals.
  ret = aclnnInplaceRoundDecimals(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRoundDecimals failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device-side memory to host-side memory, and modify based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  if (inplaceWorkspaceSize > 0) {
    aclrtFree(inplaceWorkspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
