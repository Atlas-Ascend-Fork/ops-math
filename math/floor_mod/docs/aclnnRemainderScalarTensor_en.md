# aclnnRemainderScalarTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:04:55.116Z -->

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

- Description: After broadcasting the scalar self into a tensor of the same shape as the tensor other, each element is converted to the remainder obtained by dividing it by the corresponding element of other. The result has the same sign as the divisor other, and its absolute value is less than the absolute value of other.

- Formula: The actual computation of remainder(self, other) is equivalent to the following formula:

  $$
  out_i = self - floor(self / other_i) * other_i
  $$

- Example:

  ```text
  self = 5.0   # float
  other = tensor([[-1, -2],
                  [-3, -4]]).type(int32)
  result = remainder(self, other)

  # Value of result
  # tensor([[ 0., -1.],
  #         [-1., -3.]])  float

  # For -4 in other, the calculation result is 5 - floor(5 / -4) * -4 = -3
  # The absolute value of the final result -3 is smaller than that of the original -4.
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnRemainderScalarTensorGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnRemainderScalarTensor" API to execute the computation.

```Cpp
aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(
  const aclScalar* self,
  const aclTensor* other,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnRemainderScalarTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnRemainderScalarTensorGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 175px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 120px">
  <col style="width: 130px">
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
      <td>self (aclScalar*)</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>The data type must comply with the data type deduction rules together with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>/<a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the deduced data type must be convertible to the data type of out.</td>
      <td>INT32, INT64, FLOAT16, FLOAT, DOUBLE, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Input other in the formula.</td>
      <td>Supports empty tensors.<br>The data type must comply with the data type deduction rules together with the data type of self (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>/<a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the deduced data type must be convertible to the data type of out.</td>
      <td>INT32, INT64, FLOAT16, FLOAT, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>Supports empty tensors.<br>The shape must be the same as that of other.</td>
      <td>INT32, INT64, FLOAT16, FLOAT, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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

- **Return value**

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
      <td>self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>other and out have different shapes.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed on self and other.</td>
    </tr>
    <tr>
      <td>The data type deduced from self and other is not a supported data type.</td>
    </tr>
    <tr>
      <td>The data type deduced from self and other cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The number of dimensions of other or out is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnRemainderScalarTensor

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnRemainderScalarTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. Contains the operator calculation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRemainderScalarTensor defaults to a deterministic implementation.

- When the data type of self is INT32, functionality and precision within the range [-2^24, 2^24] are prioritized.
- When other is 0 and the data type of self is an integer type, the result in out is self.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate code for resource initialization
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
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual
  // Set deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> outShape = {3, 3};
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<int64_t> otherHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t Self = 3;

  // Create the self aclScalar
  self = aclCreateScalar(&Self, aclDataType::ACL_INT64);
  CHECK_RET(self != nullptr, return ret);
  // Create the other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT64, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRemainderScalarTensor
  ret = aclnnRemainderScalarTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRemainderScalarTensor
  ret = aclnnRemainderScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRemainderScalarTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from device memory to host memory. Modify it according to the specific API definition
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it according to the specific API definition
  aclDestroyScalar(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition
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
