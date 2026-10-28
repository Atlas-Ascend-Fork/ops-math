# aclnnSWhere

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:01:54.757Z -->

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

- Operator function: Selects elements from self or other based on a condition and returns the result (broadcast supported).
- The formula is as follows:

$$
out_i=where(self_i,other_i,condition_i)=\begin{cases}
  self_i, & \text{if condition}_i \\
  other_i, & \text{otherwise}
   \end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSWhereGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnSWhere" to execute the computation.

- `aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSWhere(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSWhereGetWorkspaceSize

- **Parameters**:
  - condition (aclTensor*, computation input): Input `condition` of the formula. It is an aclTensor on the device. The supported data types are UINT8 and BOOL. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with self and other. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The supported [data format](../../../docs/en/context/data_format.md) is ND. The number of data dimensions cannot exceed 8.

  - self (aclTensor*, computation input): Input `self` of the formula. The data type of self must satisfy the data type deduction rules with the data type of other (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). It is an aclTensor on the device. The shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with other and condition. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The supported [data format](../../../docs/en/context/data_format.md) is ND. The number of data dimensions cannot exceed 8.

    <!-- npu="950,A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128, and BFLOAT16.
    <!-- end id7 -->
    <!-- npu="910,310p" id8 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, and COMPLEX128.
    <!-- end id8 -->

  - other (aclTensor*, computation input): Input `other` of the formula. The data type of other must satisfy the data type deduction rules with the data type of self (see [deduction relationship](../../../docs/en/context/deduction_relationship.md)). It is an aclTensor on the device. The shape must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with self and condition. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The supported [data format](../../../docs/en/context/data_format.md) is ND. The number of data dimensions cannot exceed 8.

    <!-- npu="950,A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128, and BFLOAT16.
    <!-- end id9 -->
    <!-- npu="910,310p" id10 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, and COMPLEX128.
    <!-- end id10 -->

  - out (aclTensor \*, computation output): Output `out` of the formula. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. It is an aclTensor on the device. The shape must be the shape after self, other, and condition are broadcast. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The supported [data format](../../../docs/en/context/data_format.md) is ND. The number of data dimensions cannot exceed 8.

    <!-- npu="950,A3,910b" id11 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128, and BFLOAT16.
    <!-- end id11 -->
    <!-- npu="910,310p" id12 -->
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The supported data types include FLOAT, INT32, UINT64, INT64, UINT32, FLOAT16, UINT16, INT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, and COMPLEX128.
    <!-- end id12 -->

  - workspaceSize (uint64_t \*, output parameter): Returns the size of the workspace to be requested on the device.
  - executor(aclOpExecutor\*\*, output parameter): Returns the op executor, which contains the computation process of the operator.

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 300px">
  <col style="width: 136px">
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
      <td>Passed self, other, condition, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and dimensions of self, other, or condition are out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and other cannot be deduced.</td>
    </tr>
    <tr>
      <td>Broadcast deduction of self, other, or condition fails, or the shape of the broadcast result is different from that of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnSWhere

- **Parameters**:

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnSWhereGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
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
  - aclnnSWhere defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_s_where.h"

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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> conditionShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* conditionDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* condition = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 0, 0, 0, 0, 0, 0, 7};
  std::vector<float> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int8_t> conditionHostData = {false,false,false,false,true,true,true,true};
  std::vector<float> outHostData = {10, 10, 10, 10, 10, 10, 10, 10};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the condition aclTensor.
  ret = CreateAclTensor(conditionHostData, conditionShape, &conditionDeviceAddr, aclDataType::ACL_BOOL, &condition);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call CANN operator library APIs
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnSWhere first-phase API
  ret = aclnnSWhereGetWorkspaceSize(condition, self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSWhereGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnSWhere second-phase API
  ret = aclnnSWhere(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSWhere failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to the host. Modify it based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(condition);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(conditionDeviceAddr);
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
