# aclnnPowTensorScalar&aclnnInplacePowTensorScalar

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:13:13.423Z -->

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
  out_i = self_i^{exponent_i}
  $$

## Function Prototype

aclnnPowTensorScalar and aclnnInplacePowTensorScalar implement the same function. The differences are as follows:

- aclnnPowTensorScalar: Requires creating a new output tensor object to store the computation result.

- aclnnInplacePowTensorScalar: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnPowTensorScalarGetWorkspaceSize" or "aclnnInplacePowTensorScalarGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnPowTensorScalar" or "aclnnInplacePowTensorScalar" to execute the computation.

```Cpp
aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(
  const aclTensor*  self,
  const aclScalar*  exponent,
  const aclTensor*  out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnPowTensorScalar(
  void*                workspace,
  uint64_t             workspaceSize,
  aclOpExecutor*       executor,
  const aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(
  const aclTensor*  self,
  const aclScalar*  exponent,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnInplacePowTensorScalar(
  void*                workspace,
  uint64_t             workspaceSize,
  aclOpExecutor*       executor,
  aclrtStream          stream)
```

## aclnnPowTensorScalarGetWorkspaceSize

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
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>The data type must be convertible to the derived data type of exponent.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, BOOL, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>exponent (aclScalar*)</td>
      <td>Input</td>
      <td>Input exponent in the formula.</td>
      <td>The data type and the data type of self cannot be BOOL at the same time.<br>If the derived data type of self and exponent is an integer type, exponent must be greater than or equal to 0.<br>The value of exponent must be within the value range of the derived data type of self and exponent. (See <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>)</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, BOOL, UINT8, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (const aclTensor*)</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>The shape must be the same as that of self, and the data type must be convertible from the data type derived from the data types of self and exponent.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BOOL, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
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

- **Return Value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
      <td>self, exponent, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of self, exponent, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of self is greater than 8 dimensions.</td>
    </tr>
    <tr>
      <td>self and exponent do not satisfy the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of out.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The value of exponent is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnPowTensorScalar

- **Parameters:**

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
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnPowTensorScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, including the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplacePowTensorScalarGetWorkspaceSize

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
        <td>self (const aclTensor*)</td>
        <td>Input</td>
        <td>Input self/out in the formula.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>exponent (aclScalar*)</td>
        <td>Input</td>
        <td>Input exponent in the formula.</td>
        <td>When the derived data types of selfRef and exponent are integer, exponent must be greater than or equal to 0.<br>The value of exponent must be within the value range of the derived data types of selfRef and exponent. (See <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>)</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, INT16, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus. For details about the return status code, see [aclnn Error Code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input selfRef or exponent is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of selfRef and exponent are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of selfRef has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>selfRef and exponent do not meet the data type derivation rules.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of selfRef.</td>
    </tr>
    <tr>
      <td>The value of exponent is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplacePowTensorScalar

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
        <td>Address of the workspace memory requested on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device, obtained by the first-phase API aclnnInplacePowTensorScalarGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- INT32 integer computation times out in scenarios outside the following ranges:

  | shape  | exponent_value|
  |----|----|
  |<=100,000 (one hundred thousand) |-200,000,000~200,000,000 (two hundred million)|
  |<=1,000,000 (one million) |-20,000,000~20,000,000 (twenty million)|
  |<=10,000,000 (ten million) |-2,000,000~2,000,000 (two million)|
  |<=100,000,000 (one hundred million) |-200,000~200,000 (two hundred thousand)|
  |<=1,000,000,000 (one billion) |-20,000~20,000 (twenty thousand)|

- Deterministic computation: aclnnPowTensorScalar&aclnnInplacePowTensorScalar default to a deterministic implementation.

<!-- npu="910,310p" id8 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: In this scenario, if the calculated result exceeds the value range of the specified data type, the boundary value of that data type is returned as the result.
<!-- end id8 -->

- When exponent = 2, the square operator is called. If the input self is of int8 type, precision is guaranteed only when the result falls within the range of (-2048, 1920).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"

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
  // Call aclrtMalloc to apply for memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* exponent = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float exponentVal = 4.1f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the exponent aclScalar.
  exponent = aclCreateScalar(&exponentVal, aclDataType::ACL_FLOAT);
  CHECK_RET(exponent != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // aclnnPowTensorScalar API call example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowTensorScalar.
  ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowTensorScalar.
  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);

  // aclnnInplacePowTensorScalar API call example
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplacePowTensorScalar.
  ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplacePowTensorScalar.
  ret = aclnnInplacePowTensorScalar(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePowTensorScalar failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host side. Modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnPowTensorScalar result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplacePowTensorScalar result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(exponent);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
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
