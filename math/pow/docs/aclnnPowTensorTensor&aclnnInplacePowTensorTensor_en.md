# aclnnPowTensorTensor&aclnnInplacePowTensorTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:17:18.154Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Each element of exponent is used as the exponent for the corresponding element of input to perform the power calculation.

- Formula:

  $$
  out_i = x_i^{exponent_i}
  $$

- Operator constraint: For INT32 integer computation scenarios outside the following range, a timeout occurs.

  | shape  | exponent_value|
  |----|----|
  |<=100,000 (one hundred thousand) |-200,000,000~200,000,000 (two hundred million)|
  |<=1,000,000 (one million) |-20,000,000~20,000,000 (twenty million)|
  |<=10,000,000 (ten million) |-2,000,000~2,000,000 (two million)|
  |<=100,000,000 (one hundred million) |-200,000~200,000 (two hundred thousand)|
  |<=1,000,000,000 (one billion) |-20,000~20,000 (twenty thousand)|

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnPowTensorTensorGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnPowTensorTensor" to execute the computation.

```Cpp
aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(
  const aclTensor*  self,
  const aclTensor*  exponent,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnPowTensorTensor(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

```Cpp
aclnnStatus aclnnInplacePowTensorTensorGetWorkspaceSize(
  const aclTensor*  self,
  const aclTensor*  exponent,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnInplacePowTensorTensor(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

## aclnnPowTensorTensorGetWorkspaceSize

- **Parameters**:

  <table style="table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
  <col style="width: 124px">
  <col style="width: 138px">
  <col style="width: 151px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>x in the formula. The base of the pow operation.</td>
      <td>The data types of self and exponent cannot be both BOOL. The shape of self must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with exponent.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>exponent (aclTensor*)</td>
      <td>Input</td>
      <td>exponent in the formula. The exponent of the pow operation.</td>
      <td>The data types of self and exponent cannot be BOOL at the same time, and the shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT16, BOOL, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>out in the formula. Stores the computation result.</td>
      <td>The data type must be a data type that self and exponent can be deduced and converted to (see the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>). The shape must be the shape after self and exponent <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BOOL, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
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

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type BFLOAT16 is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 123px">
  <col style="width: 741px">
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
      <td>self, exponent, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of self and exponent are out of the supported range.</td>
    </tr>
    <tr>
      <td>Data type deduction fails for self and exponent.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of self and exponent cannot be broadcast.</td>
    </tr>
    <tr>
      <td>self and exponent are both of the bool data type.</td>
    </tr>
  </tbody>
  </table>

## aclnnPowTensorTensor

- **Parameters**:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnPowTensorTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation process of the operator.</td>
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

## aclnnInplacePowTensorTensorGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
  <col style="width: 124px">
  <col style="width: 138px">
  <col style="width: 151px">
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
      <td>input|output</td>
      <td>x and out in the formula. Base of the pow operation. The calculation result is written back to this tensor.</td>
      <td>The data type is the type that self and exponent can be converted to after deduction. The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with exponent.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>exponent (aclTensor*)</td>
      <td>input</td>
      <td>exponent in the formula. Exponent of the pow operation.</td>
      <td>The data type must be a data type that self and exponent can be converted to after deduction (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p,310b" id8 -->
  - <term>Atlas training products</term>, <term>Atlas 200I/500 A2 inference products</term>, and <term>Atlas inference products</term>: BFLOAT16 is not supported.
  <!-- end id8 -->

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
      <td>self or exponent is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of self and exponent are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and exponent exceed 8 dimensions.</td>
    </tr>
    <tr>
      <td>self and exponent do not meet the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The shapes of self and exponent cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of self and exponent after broadcast is not equal to the shape of self.</td>
    </tr>
    <tr>
      <td>self and exponent are both of bool data type.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplacePowTensorTensor

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
        <td>workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>workspace size allocated on the Device side, obtained by the first-phase API aclnnInplacePowTensorTensorGetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPowTensorTensor&aclnnInplacePowTensorTensor default to a deterministic implementation.

<!-- npu="910,310p" id9 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: In this scenario, if the calculated result exceeds the value range of the specified data type, the boundary value of that data type is returned as the result.
<!-- end id9 -->

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnPowTensorTensor Sample Code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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
  // Call aclrtMemcpy to copy data from the host to the device memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* exp = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the exp aclTensor.
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowTensorTensor.
  ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowTensorTensor.
  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from the device side to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(exp);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(expDeviceAddr);
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

**aclnnInplacePowTensorTensor sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

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
  // Boilerplate. Resource initialization.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> expShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* expDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* exp = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> expHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the exp aclTensor.
  ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, aclDataType::ACL_FLOAT, &exp);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace the API name with the specific one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplacePowTensorTensor.
  ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exp, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplacePowTensorTensor.
  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),         selfRefDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the specific API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(exp);

  // 7. Release the device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(expDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
