# aclnnXLogYTensor&aclnnInplaceXLogYTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:34:31.526Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Operator function: Computes the result of self * log(other).
- Formula:

  $$
  out_i = self_i * log(other_i)
  $$

## Function Prototype

- aclnnXLogYTensor and aclnnInplaceXLogYTensor implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnXLogYTensor: Creates a new output tensor object to store the computation result.
  - aclnnInplaceXLogYTensor: Stores the computation result directly in the memory of the input tensor without creating a new output tensor object.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnXLogYTensorGetWorkspaceSize" or "aclnnInplaceXLogYTensorGetWorkspaceSize" first to obtain the input parameter and calculate the required workspace size based on the computation process. Then call "aclnnXLogYTensor" or "aclnnInplaceXLogYTensor" to perform the computation.
  - `aclnnStatus aclnnXLogYTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnXLogYTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceXLogYTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceXLogYTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnXLogYTensorGetWorkspaceSize

- **Parameters**

  - self(aclTensor*, computation input): Input `self` in the formula, an aclTensor on the device. The data type of self and other must satisfy the data type derivation rule (see [Deduction Relationship](../../../docs/en/context/deduction_relationship.md)), and the shape of self must comply with the [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md) with other. [Non-contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported, and the [Data Format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id7 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id7 -->

  - other(aclTensor*, computation input): Input `other` in the formula, an aclTensor on the device. The data type of other and self must satisfy the data type derivation rule (see [Deduction Relationship](../../../docs/en/context/deduction_relationship.md)), and the shape of other must comply with the [Broadcast Relationship](../../../docs/en/context/broadcast_relationship.md) with self. [Non-contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported, and the [Data Format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id8 -->

  - out(aclTensor \*, computation output): `out` in the formula, an aclTensor on the device. The shape must be the broadcast shape of self and other. [Non-contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported, and the [Data Format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id9 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id9 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
  - executor(aclOpExecutor \*\*, output parameter): Returns the op executor, which contains the operator computation process.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 294px">
  <col style="width: 134px">
  <col style="width: 721px">
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
      <td>self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of self or other is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type derived from self and other cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast, or the broadcast result is different from the shape of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnXLogYTensor

- **Parameters**

  - workspace(void \*, input parameter): The workspace memory address requested on the Device.
  - workspaceSize(uint64_t, input parameter): The size of the workspace requested on the Device, obtained by the first-phase API aclnnXLogYTensorGetWorkspaceSize.
  - executor(aclOpExecutor \*, input parameter): The op executor, including the operator computation flow.
  - stream (aclrtStream, input parameter): Stream on which the task is executed.

- **Return value**

  aclnnStatus: The return status code. For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceXLogYTensorGetWorkspaceSize

- **Parameters**

  - selfRef(aclTensor \*, computation input/output): Input `self/out` in the formula. Device-side aclTensor. The data type is a data type to which selfRef and other can be converted after deduction, and must satisfy the data type derivation rule with the data type of other (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with other. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are supported, and [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id10 -->

  - other(aclTensor*, computation input): Input `other` in the formula. Device-side aclTensor. The data type must satisfy the data type derivation rule with the data type of selfRef (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with selfRef. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are supported, and [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id11 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id11 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
  - executor(aclOpExecutor \*\*, output parameter): Returns the op executor, which contains the operator computation process.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 294px">
  <col style="width: 134px">
  <col style="width: 721px">
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
      <td>The passed selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of selfRef and other are not supported.</td>
    </tr>
    <tr>
      <td>The data type derived from selfRef and other cannot be converted to the type of the specified output selfRef.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceXLogYTensor

- **Parameters**

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnInplaceXLogYTensorGetWorkspaceSize.</td>
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
  - aclnnXLogYTensor&aclnnInplaceXLogYTensor default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_x_log_y_tensor.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMemcpy to copy data from the host to the device.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output tensors. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  // Create self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name
  // Example of calling the aclnnXLogYTensor API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnXLogYTensor
  ret = aclnnXLogYTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnXLogYTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnXLogYTensor
  ret = aclnnXLogYTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnXLogYTensor failed. ERROR: %d\n", ret); return ret);

  // Example of calling the aclnnInplaceXLogYTensor API
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceXLogYTensor
  ret = aclnnInplaceXLogYTensorGetWorkspaceSize(self, other, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceXLogYTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceXLogYTensor
  ret = aclnnInplaceXLogYTensor(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceXLogYTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host, and modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnXLogYTensor result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceXLogYTensor result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources, and modify it based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
