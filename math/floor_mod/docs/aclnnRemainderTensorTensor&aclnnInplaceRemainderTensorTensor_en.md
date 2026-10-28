# aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:07:57.886Z -->

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

- Description:
  After tensor self and tensor other are broadcast to a consistent shape, each element is converted to the remainder obtained by dividing it by the corresponding element of other. The result has the same sign as the divisor other, and the absolute value of the result is less than the absolute value of other.
  The actual computation of remainder(self, other) is equivalent to the following formula:

  $$
  out_i = self_i - floor(self_i / other_i) * other_i
  $$

- Example:

  ```text
  self = tensor([[-1, -2],
                 [-3, -4]]).type(int64)
  other = tensor([-3, -3]).type(float16)

  result = remainder(self, other)

  # Value of result
  # tensor([[-1., -2.],
  #         [-0., -1.]], dtype=float16)

  # First, other is broadcast to the same shape as self, becoming [[-3, -3], [-3, -3]], and then the computation is performed.
  # For the element -3 in self, the calculation result is (-3) % (-3) = 0
  # It can be seen that the absolute value of the final result 0 is less than that of the original -3.
  ```

## Function Prototype

- aclnnRemainderTensorTensor and aclnnInplaceRemainderTensorTensor implement the same functionality. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnRemainderTensorTensor: Creates a new output tensor object to store the computation result.
  - aclnnInplaceRemainderTensorTensor: Stores the computation result directly in the memory of the input tensor without creating a new output tensor object.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnRemainderTensorTensorGetWorkspaceSize" or "aclnnInplaceRemainderTensorTensorGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnRemainderTensorTensor" or "aclnnInplaceRemainderTensorTensor" to execute the computation.

  - `aclnnStatus aclnnRemainderTensorTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnRemainderTensorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  - `aclnnStatus aclnnInplaceRemainderTensorTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceRemainderTensorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRemainderTensorTensorGetWorkspaceSize

- **Parameters:**

  - self(aclTensor*, calculation input): `self` in the formula. Its data type and the data type of other need to satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md), and the data type after mutual derivation must be within the following range. Its shape needs to satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with other. Supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), [data format](../../../docs/en/context/data_format.md) supports ND, and data dimensions of 8 dimensions or more are not supported.

    <!-- npu="950,A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type after mutual derivation is INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16.
    <!-- end id7 -->
    <!-- npu="910,310p" id8 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type after mutual derivation is INT32, INT64, FLOAT16, FLOAT, or DOUBLE.
    <!-- end id8 -->

  - other(aclTensor*, calculation input): `other` in the formula. Its data type and the data type of self need to satisfy the [deduction relationship](../../../docs/en/context/deduction_relationship.md), and the data type after mutual derivation must be within the following range. Its shape needs to satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with self. Supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), [data format](../../../docs/en/context/data_format.md) supports ND, and data dimensions of 8 dimensions or more are not supported.

    <!-- npu="950,A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type after mutual derivation is INT32, INT64, FLOAT16, FLOAT, DOUBLE, or BFLOAT16.
    <!-- end id9 -->
    <!-- npu="910,310p" id10 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type after mutual derivation is INT32, INT64, FLOAT16, FLOAT, or DOUBLE.
    <!-- end id10 -->

  - out(aclTensor*, calculation output): The output `out` in the formula. Its data type needs to be the data type that can be converted after the derivation of self and other. Its shape needs to be the shape after self and other are broadcast. Supports [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md), [data format](../../../docs/en/context/data_format.md) supports ND, and data dimensions of 8 dimensions or more are not supported.

    <!-- npu="950,A3,910b" id11 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type supports UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, and BFLOAT16.
    <!-- end id11 -->
    <!-- npu="910,310p" id12 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type supports UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE, COMPLEX64, and COMPLEX128.
    <!-- end id12 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
  - executor (aclOpExecutor**, output parameter): Returns the op executor, which includes computation.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 738px">
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
      <td>Null pointer passed for self, other, or out.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of self and other cannot be derived.</td>
    </tr>
    <tr>
      <td>The data type derived from self and other is not among the supported data types.</td>
    </tr>
    <tr>
      <td>The data type derived from self and other cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of self and other after broadcast is inconsistent with the shape of out.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self, other, or out is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnRemainderTensorTensor

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
      <td>Memory address of the workspace applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnRemainderTensorTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceRemainderTensorTensorGetWorkspaceSize

- **Parameters:**

  - selfRef(aclTensor*, computation input|computation output): Input/output tensor. The data type of selfRef and the data type of other need to satisfy the data type deduction rules (see also [deduction relationship](../../../docs/en/context/deduction_relationship.md)) and need to be data types that are convertible after mutual derivation. The shape needs to satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with other, and the shape after broadcasting needs to be the same as the shape of selfRef. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../../docs/en/context/data_format.md) supports ND. Data dimensions of 8 dimensions or more are not supported.

    <!-- npu="950,A3,910b" id13 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type supports INT32, INT64, FLOAT16, FLOAT, DOUBLE, and BFLOAT16.
    <!-- end id13 -->
    <!-- npu="910,310p" id14 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type supports INT32, INT64, FLOAT16, FLOAT, and DOUBLE.
    <!-- end id14 -->

  - other(aclTensor*, computation input): Input `other` in the formula. The data type of selfRef and the data type of other need to satisfy the data type deduction rules (see also [deduction relationship](../../../docs/en/context/deduction_relationship.md)). The shape needs to satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with selfRef. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported, and the [data format](../../../docs/en/context/data_format.md) supports ND. Data dimensions of 8 dimensions or more are not supported.

    <!-- npu="950,A3,910b" id15 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The data type supports INT32, INT64, FLOAT16, FLOAT, DOUBLE, and BFLOAT16.
    <!-- end id15 -->
    <!-- npu="910,310p" id16 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type supports INT32, INT64, FLOAT16, FLOAT, and DOUBLE.
    <!-- end id16 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.

  - executor (aclOpExecutor**, output parameter): Returns the op executor, which includes computation.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 738px">
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
      <td>Null pointer passed for selfRef or other.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of selfRef and other cannot be deduced.</td>
    </tr>
    <tr>
      <td>The data type derived from selfRef and other is not among the supported data types.</td>
    </tr>
    <tr>
      <td>The data type derived from selfRef and other cannot be converted to the type of selfRef.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of selfRef and other after broadcasting is inconsistent with the shape of selfRef.</td>
    </tr>
    <tr>
      <td>The number of dimensions of selfRef and other is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceRemainderTensorTensor

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnInplaceRemainderTensorTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**
  - aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor default to a deterministic implementation.

- When the data type of self is INT32, functionality and precision within the range [-2^24, 2^24] are prioritized.
- When other is 0 and the data type of self is an integer type, the result in out is self.

## Example

The example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**Sample code for aclnnRemainderTensorTensor:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_remainder.h"

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
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. The construction needs to be customized based on the API.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> outShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRemainderTensorTensor.
  ret = aclnnRemainderTensorTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRemainderTensorTensor.
  ret = aclnnRemainderTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device-side memory to host-side memory. Modify it according to the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify them according to the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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

**aclnnInplaceRemainderTensorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_remainder.h"

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
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to apply for device-side memory.
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

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfRefShape = {3, 3};
  std::vector<int64_t> otherShape = {3, 3};
  void* selfRefDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* other = nullptr;
  std::vector<int64_t> selfRefHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT64, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceRemainderTensorTensor.
  ret = aclnnInplaceRemainderTensorTensorGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRemainderTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnInplaceRemainderTensorTensor second-phase API.
  ret = aclnnInplaceRemainderTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRemainderTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the value of the output and copy the result on the device side to the host side. The code needs to be modified according to the specific API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. The code needs to be modified according to the specific API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(other);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
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
