# aclnnLogicalOr&aclnnInplaceLogicalOr

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:46:50.295Z -->

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

- Description: Performs the logical OR operation on the elements of the given input tensors. When the two input tensors are of non-bool type, 0 is treated as False and non-zero is treated as True.
- Formula:

$$
out[i]=
\begin{cases}
1, & \text{if } self[i] \neq 0 \text{ or } other[i] \neq 0 \\
0, & \text{otherwise}
\end{cases}
$$

## Function Prototype

- aclnnLogicalOr and aclnnInplaceLogicalOr implement the same functionality. The usage differences are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnLogicalOr: A new output tensor object needs to be created to store the computation result.
  - aclnnInplaceLogicalOr: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnLogicalOrGetWorkspaceSize" or "aclnnInplaceLogicalOrGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnLogicalOr" or "aclnnInplaceLogicalOr" to execute the computation.

```Cpp
aclnnStatus aclnnLogicalOrGetWorkspaceSize(
  const aclTensor*     self,
  const aclTensor*     other,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnLogicalOr(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceLogicalOrGetWorkspaceSize(
  aclTensor*           selfRef,
  const aclTensor*     other,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnInplaceLogicalOr(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnLogicalOrGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">shape needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Tensor to compute the logical OR operation.</td>
      <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor.</td>
      <td class="tg-0pky">        <ul>
          <li>The shape is the same as the shape of self and other after broadcasting.</li>
          <li>The data type must be a data type convertible from BOOL (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be requested on the Device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b,910,310p" id8 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: The out data type does not support UINT16, UINT32, or UINT64.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1144px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 733px">
  </colgroup>
  <thead>
    <tr>
      <th>return value</th>
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
      <td>The data type of self, other, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The dimensions of self, other, and out are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnLogicalOr

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnLogicalOrGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLogicalOrGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">selfRef (aclTensor*)</td>
      <td class="tg-0pky">Input|Output</td>
      <td class="tg-0pky">Input|Output tensor.</td>
      <td class="tg-0pky">        <ul>
          <li>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li>
          <li>The shape after broadcast is the same as the shape of selfRef.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Tensor for the OR operation.</td>
      <td class="tg-0pky">shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id10 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id10 -->

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
      <td>selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of selfRef or other are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of selfRef is not the shape of selfRef and other after broadcast.</td>
    </tr>
    <tr>
      <td>The dimensions of selfRef or other are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceLogicalOr

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnInplaceLogicalOrGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
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

## Constraints

- Deterministic computation:
  - aclnnLogicalOr & aclnnInplaceLogicalOr default to a deterministic implementation.

## Example

The example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnLogicalOr Sample Code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logical_or.h"

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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. You need to construct them based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 0, 5, 0, 0};
  std::vector<float> otherHostData = {1, 1, 1, 0, 0, 2, 3, 0};
  std::vector<float> outHostData(8, 0);

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLogicalOr.
  ret = aclnnLogicalOrGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogicalOrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLogicalOr.
  ret = aclnnLogicalOr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogicalOr failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
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

**aclnnInplaceLogicalOr sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logical_or.h"

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
  // Boilerplate: resource initialization.
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
  // Call aclrtMemcpy to copy host data to device memory.
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
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> selfRefShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfRefDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* other = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 0, 5, 0, 0};
  std::vector<float> otherHostData = {1, 1, 1, 0, 0, 2, 3, 0};

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceLogicalOr.
  ret = aclnnInplaceLogicalOrGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLogicalOrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceLogicalOr.
  ret = aclnnInplaceLogicalOr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLogicalOr failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from the device side to the host side.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(other);

  // 7. Release device resources. Modify this step according to the specific API definition.
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
