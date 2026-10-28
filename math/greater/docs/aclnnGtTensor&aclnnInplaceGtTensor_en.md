# aclnnGtTensor&aclnnInplaceGtTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:11:14.571Z -->

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

- Description: Determines whether elements in the self tensor are greater than elements in the other tensor. Returns a tensor where the result is True(1) if self > other; otherwise False(0).
- Formula:

  $$
  out = (self_i > other_i)  ?  [True] : [False]
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnGtTensorGetWorkspaceSize" or "aclnnInplaceGtTensorGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnGtTensor" or "aclnnInplaceGtTensor" to execute the computation.

```Cpp
aclnnStatus aclnnGtTensorGetWorkspaceSize(
  const aclTensor*  self,
  const aclTensor*  other,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnGtTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceGtTensorGetWorkspaceSize(
  const aclTensor*  selfRef,
  const aclTensor*  other,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnInplaceGtTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGtTensorGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 291px">
  <col style="width: 286px">
  <col style="width: 284px">
  <col style="width: 120px">
  <col style="width: 141px">
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
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor that participates in the Greater size comparison, corresponding to self in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor that participates in the greater size comparison, corresponding to other in the formula.</td>
      <td>Tensor type. The data type must satisfy the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">data type deduction rules</a> with self, and the shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with the shape of self.</td>
      <td>DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output of the size comparison, corresponding to out in the formula.</td>
      <td>The data type must be one that BOOL can be converted to.</td>
      <td>DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
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
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
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
      <td>self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self, other, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of self, other, or out is greater than 8.</td>
    </tr>
    <tr>
      <td>The data types of self and other cannot perform <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction</a>.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot perform broadcast.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the shape after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</td>
    </tr>
  </tbody>
  </table>

## aclnnGtTensor

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
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
      <td>Memory address of the workspace allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnGtTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation process.</td>
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

## aclnnInplaceGtTensorGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 291px">
  <col style="width: 286px">
  <col style="width: 284px">
  <col style="width: 120px">
  <col style="width: 141px">
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
      <th>Dim (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input tensor that participates in the Greater size comparison computation. It is also the output after the Greater comparison, corresponding to self and out in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor that participates in the greater size comparison computation, corresponding to other in the formula.</td>
      <td>Tensor. The data type must satisfy the data type deduction rules with the data type of selfRef.</td>
      <td>DOUBLE, FLOAT16, FLOAT32, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, BOOL, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>Returns the executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id8 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 272px">
  <col style="width: 114px">
  <col style="width: 764px">
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
      <td>selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of selfRef and other are not supported.</td>
    </tr>
    <tr>
      <td>The data types of selfRef and other cannot be deduced.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</td>
    </tr>
    <tr>
      <td>The shape of selfRef and other after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> is not equal to the shape of selfRef.</td>
    </tr>
    <tr>
      <td>The dimension of selfRef or other is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceGtTensor

- **Parameters:**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 873px">
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
      <td>Workspace memory address on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size on the device, obtained by calling the first-phase API aclnnInplaceGtTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGtTensor&aclnnInplaceGtTensor default to a deterministic implementation.

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnGtTensor example code**:

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gt_tensor.h"

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
  // Boilerplate for resource initialization.
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

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct GtTensorData {
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 1, 2, 3, 4, 5, 6};
  std::vector<char> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
};

int PrepareAndExecuteGtTensor(aclrtStream stream, GtTensorData& data, void*& workspaceAddr, uint64_t& workspaceSize) {
  auto ret = 0;
  // Create the self aclTensor.
  ret = CreateAclTensor(data.selfHostData, data.selfShape, &data.selfDeviceAddr, aclDataType::ACL_DOUBLE, &data.self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(data.otherHostData, data.otherShape, &data.otherDeviceAddr, aclDataType::ACL_DOUBLE, &data.other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(data.outHostData, data.outShape, &data.outDeviceAddr, aclDataType::ACL_BOOL, &data.out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Call the CANN operator library API.
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGtTensor.
  ret = aclnnGtTensorGetWorkspaceSize(data.self, data.other, data.out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGtTensor.
  ret = aclnnGtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGtTensor failed. ERROR: %d\n", ret); return ret);

  return 0;
}

int HandleGtTensorResult(aclrtStream stream, const GtTensorData& data, void* workspaceAddr, uint64_t workspaceSize) {
  auto ret = 0;
  // Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result from device memory to host memory.
  auto size = GetShapeSize(data.outShape);
  std::vector<char> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), data.outDeviceAddr, size * sizeof(char),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // Destroy aclTensor and aclScalar.
  aclDestroyTensor(data.self);
  aclDestroyTensor(data.other);
  aclDestroyTensor(data.out);

  // Release device resources.
  aclrtFree(data.selfDeviceAddr);
  aclrtFree(data.otherDeviceAddr);
  aclrtFree(data.outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return 0;
}

int ExecuteGtTensorOperator(aclrtStream stream) {
  GtTensorData data;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;

  auto ret = PrepareAndExecuteGtTensor(stream, data, workspaceAddr, workspaceSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = HandleGtTensorResult(stream, data, workspaceAddr, workspaceSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return 0;
}

int main() {
  // Initialize device/stream. For details, see the ACL API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Execute the GtTensor operation.
  ret = ExecuteGtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteGtTensorOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and finalize ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**Example code for aclnnInplaceGtTensor**:

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gt_tensor.h"

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
  // Boilerplate code, resource initialization.
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

  // Call aclrtMemcpy to copy data from the host to device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> otherHostData = {0, 1, 1, 2, 3, 4, 5, 6};

  // Create self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_DOUBLE, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with a specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceGtTensor.
  ret = aclnnInplaceGtTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceGtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceGtTensor.
  ret = aclnnInplaceGtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceGtTensor failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device-side memory to the host side. Modify it according to the definition of the specific API.
  auto size = GetShapeSize(selfShape);
  std::vector<char> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify it according to the definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(other);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
