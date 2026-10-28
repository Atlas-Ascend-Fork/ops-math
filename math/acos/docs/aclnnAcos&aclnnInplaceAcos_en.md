# aclnnAcos&aclnnInplaceAcos

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:59:30.332Z -->

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

- Description: Performs the arccosine operation on each element of the input matrix and outputs the result.

- Formula:

  $$
  out_{i}=cos^{-1}(input_{i})
  $$

## Function Prototype

- aclnnAcos and aclnnInplaceAcos implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnAcos: Create a new output tensor object to store the computation result.
  - aclnnInplaceAcos: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnAcosGetWorkspaceSize" or "aclnnInplaceAcosGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the computation process of the operator, and then call "aclnnAcos" or "aclnnInplaceAcos" to execute the computation.

```cpp
aclnnStatus aclnnAcosGetWorkspaceSize(
  const aclTensor*    input,
  aclTensor*          out,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnAcos(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```cpp
aclnnStatus aclnnInplaceAcosGetWorkspaceSize(
  aclTensor*          inputRef,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnInplaceAcos(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnAcosGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 140px">
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
        <td>input (aclTensor*)</td>
        <td>input</td>
        <td>-</td>
        <td>When the type is INT8, INT16, INT32, INT64, UINT8, or BOOL, it is converted to FLOAT for computation and the output is of the FLOAT type. The shape must be the same as that of out.</td>
        <td>INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>-</td>
        <td>shape must be the same as that of input.</td>
        <td>FLOAT, FLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the Device.</td>
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

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: The input and out data types additionally support BFLOAT16.
  <!-- end id7 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The data types of input and out additionally support BFLOAT16.
  <!-- end id9 -->

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
      <td>The input or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of input or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of input and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input or out is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnAcos

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
        <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnAcosGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
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

## aclnnInplaceAcosGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 140px">
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
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>inputRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Input/output tensor, corresponding to input and out in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3" id8 -->
  - <term>Atlas A3 products</term>: The inputRef data type additionally supports BFLOAT16.
  <!-- end id8 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: The inputRef data type has additional support for BFLOAT16.
  <!-- end id10 -->

- **Return value**

  aclnnStatus: return status, see also [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The inputRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of inputRef is out of the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of inputRef is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAcos

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
        <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnInplaceAcosGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, which contains the operator computation flow.</td>
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
  - aclnnAcos&aclnnInplaceAcos defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_acos.h"

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
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction is custom based on the API interface.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {1, -1, 0, 0.5, -1.732/2, 12, NAN, -INFINITY};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAcos API call example
  // 3. Call the CANN operator library API
  // Call the aclnnAcos first-phase API
  ret = aclnnAcosGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcosGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnAcos second-phase API
  ret = aclnnAcos(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAcos failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("%ld acos(%lf) = %lf\n", i, selfHostData[i], resultData[i]);
  }

  //aclnnInplaceAcos API call example
  // 3. Call the CANN operator library API
  LOG_PRINT("\ntest aclnnInplaceAcos\n");
  // Call the aclnnInplaceAcos first-phase API
  ret = aclnnInplaceAcosGetWorkspaceSize(self, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAcosGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnInplaceAcos.
  ret = aclnnInplaceAcos(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAcos failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Wait synchronously until the task is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device memory to the host side.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("%ld acos(%f) = %f\n", i, selfHostData[i], resultData[i]);
  }


  // 6. Release the aclTensor, which needs to be modified based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release the Device resources, which need to be modified based on the specific API definition.
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
