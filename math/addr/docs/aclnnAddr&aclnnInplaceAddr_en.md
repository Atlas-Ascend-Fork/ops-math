# aclnnAddr&aclnnInplaceAddr

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:15:34.486Z -->

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

- Description: Computes the outer product of one-dimensional vectors vec1 and vec2 to obtain a two-dimensional matrix, multiplies the outer product result matrix by a factor, adds it to self multiplied by a factor, and outputs the result.

- Formula:
  $$
  \text{out} = \beta\ \text{self} + \alpha\ (\text{vec1} \otimes\text{vec2})
  $$

## Function Prototype

- aclnnAddr and aclnnInplaceAddr implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnAddr: Create a new output tensor object to store the computation result.
  - aclnnInplaceAddr: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAddrGetWorkspaceSize" or "aclnnInplaceAddrGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnAddr" or "aclnnInplaceAddr" API to perform the computation.

```cpp
aclnnStatus aclnnAddrGetWorkspaceSize(
  const aclTensor*    self,
  const aclTensor*    vec1,
  const aclTensor*    vec2,
  const aclScalar*    betaOptional,
  const aclScalar*    alphaOptional,
  aclTensor*          out,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnAddr(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```cpp
aclnnStatus aclnnInplaceAddrGetWorkspaceSize(
  aclTensor*          selfRef,
  const aclTensor*    vec1,
  const aclTensor*    vec2,
  const aclScalar*    betaOptional,
  const aclScalar*    alphaOptional,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnInplaceAddr(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnAddrGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1570px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 120px">
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
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Outer product extension matrix.</td>
        <td>The shape dimensions must not exceed 2 and must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with vec1 and vec2.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vec1 (aclTensor*)</td>
        <td>Input</td>
        <td>The first vector of the outer product input, a one-dimensional vector.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vec2 (aclTensor*)</td>
        <td>Input</td>
        <td>Second vector of the outer product input, which is a one-dimensional vector.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>betaOptional (aclScalar*)</td>
        <td>Input</td>
        <td>Scale factor of the outer product extension matrix, that is, β in the formula.</td>
        <td>If betaOptional is of the bool type, the data type of self/vec1/vec2 can only be bool. If self/vec1/vec2 is an integer, betaOptional and alphaOptional cannot be floating-point.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>alphaOptional (aclScalar*)</td>
        <td>Input</td>
        <td>Outer product scaling factor, that is, α in the formula.</td>
        <td>If alphaOptional is of the bool data type, the data type of self/vec1/vec2 can only be bool; if self/vec1/vec2 is of an integer type, betaOptional and alphaOptional cannot be of floating-point types.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output result.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the device.</td>
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
    </tbody></table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

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
      <td>The tensor or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types and data formats of self, vec1, and vec2 are not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions of vec1 and vec2 are not 1, or the dimension of self exceeds 2.</td>
    </tr>
    <tr>
      <td>self cannot be extended to the shape of the outer product result of vec1 and vec2.</td>
    </tr>
    <tr>
      <td>When beta or alpha is of the bool type, the data types of self, vec1, and vec2 are not bool.</td>
    </tr>
    <tr>
      <td>When self, vec1, and vec2 are all of the integer type, bool type, or "integer + bool", beta or alpha is of the floating-point type.</td>
    </tr>
  </tbody>
  </table>

## aclnnAddr

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
        <td>Workspace memory address requested on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace memory requested on the Device side, obtained by calling the first-phase API aclnnAddrGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream used to execute tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddrGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1570px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 120px">
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
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>selfRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Outer product extension matrix and output matrix.</td>
        <td>The shape dimension is 2. Empty tensors are not supported. It must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with vec1 and vec2.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vec1 (aclTensor*)</td>
        <td>Input</td>
        <td>The first vector of the outer product input, a one-dimensional vector.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vec2 (aclTensor*)</td>
        <td>Input</td>
        <td>The second vector of the outer product input, a one-dimensional vector.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>betaOptional (aclScalar*)</td>
        <td>Input</td>
        <td>Scale factor of the outer product extension matrix, that is, β in the formula.</td>
        <td>If betaOptional is of the bool type, the data type of self/vec1/vec2 can only be bool; if self/vec1/vec2 are of integer types, betaOptional and alphaOptional cannot be of floating-point types.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>alphaOptional (aclScalar*)</td>
        <td>Input</td>
        <td>Scale factor of the outer product, that is, α in the formula.</td>
        <td>If alphaOptional is of bool type, the data type of self/vec1/vec2 can only be bool. If self/vec1/vec2 is of integer type, betaOptional and alphaOptional cannot be of floating-point type.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
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
        <td>Returns the executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
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
      <td>The tensor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types and data formats of selfRef, vec1, and vec2 are not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of vec1 or vec2 is not 1, or the dimension of selfRef is not 2.</td>
    </tr>
    <tr>
      <td>selfRef cannot be expanded to the shape of the outer product result of vec1 and vec2.</td>
    </tr>
    <tr>
      <td>When beta or alpha is of the bool type, the data type of selfRef, vec1, or vec2 is not bool.</td>
    </tr>
    <tr>
      <td>When the types of selfRef, vec1, and vec2 are all integer, bool, or "integer + bool", beta or alpha is of the floating-point type.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAddr

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
        <td>Workspace size requested on the Device, obtained by the first-phase API aclnnInplaceAddrGetWorkspaceSize.</td>
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
  - aclnnAddr&aclnnInplaceAddr defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addr.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
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
  //(Boilerplate) device/stream initialization, refer to the acl API manual.
  // Set deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> inputShape = {3, 2};
  std::vector<int64_t> vec1Shape = {3};
  std::vector<int64_t> vec2Shape = {2};
  std::vector<int64_t> outShape = {3, 2};

  void* inputDeviceAddr = nullptr;
  void* vec1DeviceAddr = nullptr;
  void* vec2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* vec1 = nullptr;
  aclTensor* vec2 = nullptr;
  aclScalar* beta = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> vec1HostData = {1, 2, 3};
  std::vector<float> vec2HostData = {4, 5};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
  float betaValue = 1.5f;
  float alphaValue = 1.5f;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec1HostData, vec1Shape, &vec1DeviceAddr, aclDataType::ACL_FLOAT, &vec1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vec2HostData, vec2Shape, &vec2DeviceAddr, aclDataType::ACL_FLOAT, &vec2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create beta and alpha scalar values.
  beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);


  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

   // Example of calling the aclnnAddr API.
   // Call the first-phase API of aclnnAddr.
  ret = aclnnAddrGetWorkspaceSize(input, vec1, vec2, beta, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnAddr.
  ret = aclnnAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddr failed. ERROR: %d\n", ret); return ret);

  // (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result on the device side to the host side. Modify the code based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // API call example of aclnnInplaceAddr
  // Call the first-phase API of aclnnInplaceAddr, where selfRef is the input and output tensor. Pass out here.
  ret = aclnnInplaceAddrGetWorkspaceSize(out, vec1, vec2, beta, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnInplaceAddr.
  ret = aclnnInplaceAddr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddr failed. ERROR: %d\n", ret); return ret);

  // (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Obtain the output value and copy the result on the device side to the host side. Modify the code based on the definition of the specific API.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // Release aclTensor and aclScalar.
  aclDestroyTensor(input);
  aclDestroyTensor(vec1);
  aclDestroyTensor(vec2);
  aclDestroyTensor(out);

  // Release device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(vec1DeviceAddr);
  aclrtFree(vec2DeviceAddr);
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
