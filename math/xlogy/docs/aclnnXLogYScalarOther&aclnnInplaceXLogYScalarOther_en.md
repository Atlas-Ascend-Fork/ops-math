# aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:35:35.030Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Operator function: Computes the result of self * log(other).
- Formula:

  $$
  out_i = self_i * log(other_i)
  $$

## Function Prototype

- aclnnXLogYScalarOther and aclnnInplaceXLogYScalarOther implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnXLogYScalarOther: Creates a new output tensor object to store the computation result.
  - aclnnInplaceXLogYScalarOther: Stores the computation result directly in the memory of the input tensor without creating a new output tensor object.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnXLogYScalarOtherGetWorkspaceSize" or "aclnnInplaceXLogYScalarOtherGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the computation process, and then call "aclnnXLogYScalarOther" or "aclnnInplaceXLogYScalarOther" to execute the computation.
  - `aclnnStatus aclnnXLogYScalarOtherGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnXLogYScalarOther(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceXLogYScalarOtherGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceXLogYScalarOther(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnXLogYScalarOtherGetWorkspaceSize

- **Parameters**

  - self(aclTensor*, computation input): Input `self` in the formula, an aclTensor on the Device. The data type of self and that of other must satisfy the data type derivation rule (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id7 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id7 -->

  - other(aclScalar*, computation input): Input `other` in the formula, an aclScalar on the Host. The data type of other and that of self must satisfy the data type derivation rule (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id8 -->

  - out(aclTensor \*, computation output): `out` in the formula, an aclTensor on the Device. out has the same shape as self. Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id9 -->
    - <term>Atlas A3 products</term>: Data types supported include FLOAT, FLOAT16, and BFLOAT16.
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
      <td>The passed-in self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of self and other are out of the supported range.</td>
    </tr>
    <tr>
      <td>The data type derived from self and other cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are different.</td>
    </tr>
  </tbody>
  </table>

## aclnnXLogYScalarOther

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnXLogYScalarOtherGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
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

## aclnnInplaceXLogYScalarOtherGetWorkspaceSize

- **Parameters**

  - selfRef(aclTensor \*, computation input|computation output): Input `self/out` in the formula, an aclTensor on the Device. The data type of selfRef must satisfy the data type derivation rule with the data type of other (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)) and be convertible to the data type derived from selfRef and other. Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md). [Data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: Data types supported include FLOAT, FLOAT16, and BFLOAT16.
    <!-- end id10 -->

  - other(aclScalar*, computation input): Input `other` in the formula, an aclScalar on the Host. The data type of other must satisfy the data type derivation rule with the data type of selfRef (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md). [Data format](../../../docs/en/context/data_format.md) supports ND.

    <!-- npu="A3" id11 -->
    - <term>Atlas A3 products</term>: Supported data types include FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    <!-- end id11 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
  - executor(aclOpExecutor \*\*, output parameter): Returns the op executor, which contains the computation process of the operator.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API validates input parameters and reports an error in the following use cases:
  Returns 161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input selfRef or other is a null pointer.
  Returns 161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of selfRef and other are not within the supported range.
                                       2. The data type derived from selfRef and other cannot be converted to the type of the specified output selfRef.
  ```

## aclnnInplaceXLogYScalarOther

- **Parameters**

  - workspace(void \*, input parameter): Pointer to the workspace memory allocated on the Device.
  - workspaceSize(uint64_t, input parameter): Size of the workspace memory allocated on the Device, obtained by the first-phase API aclnnInplaceXLogYScalarOtherGetWorkspaceSize.
  - executor(aclOpExecutor \*, input parameter): op executor, which contains the computation process of the operator.
  - stream (aclrtStream, input parameter): Stream on which the task is executed.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnXLogYScalarOther & aclnnInplaceXLogYScalarOther default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_x_log_y_scalar_other.h"

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  float otherValue = 2.0f;
  std::vector<float> outHostData(8, 0);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // Example of calling the aclnnXLogYScalarOther API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnXLogYScalarOther.
  ret = aclnnXLogYScalarOtherGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnXLogYScalarOtherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnXLogYScalarOther.
  ret = aclnnXLogYScalarOther(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnXLogYScalarOther failed. ERROR: %d\n", ret); return ret);

  // Example of calling the aclnnInplaceXLogYScalarOther API
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceXLogYScalarOther.
  ret = aclnnInplaceXLogYScalarOtherGetWorkspaceSize(self, other, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceXLogYScalarOtherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceXLogYScalarOther.
  ret = aclnnInplaceXLogYScalarOther(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceXLogYScalarOther failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate code) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device-side memory to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnXLogYScalarOther result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(size, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr, size * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceXLogYScalarOther result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclDestroyTensor(out);

  // 7.Release device resources, which needs to be modified according to the specific API definition.
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
