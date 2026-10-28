# aclnnLtTensor&aclnnInplaceLtTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:31:32.764Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Determines whether each element in the input self is less than the corresponding element in the input other, and returns a tensor of Bool type.

- Formula:

  $$
  out_i = (self_i < other_i)  ?  [True] : [False]
  $$

## Function Prototype

- aclnnLtTensor and aclnnInplaceLtTensor implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnLtTensor: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceLtTensor: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnLtTensorGetWorkspaceSize" or "aclnnInplaceLtTensorGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow. Then, call the "aclnnLtTensor" or "aclnnInplaceLtTensor" API to perform the computation.

```Cpp
aclnnStatus aclnnLtTensorGetWorkspaceSize(
  const aclTensor*     self,
  const aclTensor*     other,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnLtTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnInplaceLtTensorGetWorkspaceSize(
  const aclTensor*     selfRef,
  const aclTensor*     other,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnInplaceLtTensor(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnLtTensorGetWorkspaceSize

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
      <td class="tg-0pky">        <ul>
          <li>The data type must satisfy the data type derivation rule with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must satisfy the data type derivation rule of self (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor.</td>
      <td class="tg-0pky">        <ul>
          <li>The shape must be consistent with the shape after self and other are broadcast.</li>
          <li>The data type must be a data type that BOOL can be converted to (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">Conversion Relationship</a>).</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL, COMPLEX64, COMPLEX128</td>
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
      <td class="tg-0pky">Returns the op executor containing the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310b" id7 -->
  - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 741px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
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
      <td>The data type of self, other, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of self, other, or out is greater than 8.</td>
    </tr>
    <tr>
      <td>The data types of self and other cannot be derived.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of out is not consistent with the shape after broadcast.</td>
    </tr>
  </tbody>
  </table>

## aclnnLtTensor

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
      <td>Workspace memory address applied for on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied for on the Device. The value is obtained by the first-phase API aclnnLtTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceLtTensorGetWorkspaceSize

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
          <li>The data type must satisfy the data type derivation rules together with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li>
          <li>The shape after broadcasting must be consistent with the shape of selfRef.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must satisfy the data type derivation rule with the data type of selfRef (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li>
          <li>The shape after broadcast must be consistent with the shape of selfRef.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, INT32, UINT32, INT64, UINT64, INT16, UINT16, INT8, UINT8, DOUBLE, BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be requested on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310b" id8 -->
  - <term>Atlas 200I/500 A2 inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
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
      <td>selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of selfRef and other are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of selfRef and other cannot be derived.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of selfRef and other after broadcast is not equal to the shape of selfRef.</td>
    </tr>
    <tr>
      <td>The dimension of selfRef or other is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceLtTensor

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device side, obtained by calling the first-phase API aclnnInplaceLtTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLtTensor&aclnnInplaceLtTensor default to a deterministic implementation.

## Example

The following example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnLtTensor Example:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

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
  // Boilerplate for resource initialization
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


struct LtTensorData {
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
  std::vector<double> otherHostData = {5, 5, 5, 5, 5, 5, 5, 5};
  std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
};

int CreateInputAndOutputTensors(LtTensorData& data) {
  auto ret = 0;

  // Create the self aclTensor.
  ret = CreateAclTensor(data.selfHostData, data.selfShape, &data.selfDeviceAddr, aclDataType::ACL_DOUBLE, &data.self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(data.otherHostData, data.otherShape, &data.otherDeviceAddr, aclDataType::ACL_DOUBLE, &data.other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(data.outHostData, data.outShape, &data.outDeviceAddr, aclDataType::ACL_DOUBLE, &data.out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ret;
}

int ExecuteLtTensorComputation(aclrtStream stream, LtTensorData& data) {
  auto ret = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnLtTensor.
  ret = aclnnLtTensorGetWorkspaceSize(data.self, data.other, data.out, &data.workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  data.workspaceAddr = nullptr;
  if (data.workspaceSize > 0) {
    ret = aclrtMalloc(&data.workspaceAddr, data.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnLtTensor.
  ret = aclnnLtTensor(data.workspaceAddr, data.workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLtTensor failed. ERROR: %d\n", ret); return ret);

  // Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  return ret;
}

int ProcessAndPrintResults(const LtTensorData& data) {
  auto ret = 0;
  auto size = GetShapeSize(data.outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), data.outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }
  return ret;
}

void ReleaseResources(LtTensorData& data) {
  // Release aclTensor and aclScalar.
  aclDestroyTensor(data.self);
  aclDestroyTensor(data.other);
  aclDestroyTensor(data.out);

  // Release device resources.
  aclrtFree(data.selfDeviceAddr);
  aclrtFree(data.otherDeviceAddr);
  aclrtFree(data.outDeviceAddr);
  if (data.workspaceSize > 0) {
    aclrtFree(data.workspaceAddr);
  }
}

int ExecuteLtTensorOperator(aclrtStream stream) {
  LtTensorData data;

  // Create input and output tensors.
  auto ret = CreateInputAndOutputTensors(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Execute the LtTensor operator.
  ret = ExecuteLtTensorComputation(stream, data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Process and print the result.
  ret = ProcessAndPrintResults(data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Release resources.
  ReleaseResources(data);

  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Execute the LtTensor operation.
  ret = ExecuteLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteLtTensorOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and finalize ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

**Sample code for aclnnInplaceLtTensor:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lt_tensor.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
  if (!(cond)) {                     \
    return_expr;                     \
  }                                  \
 } while (0)

#define LOG_PRINT(message, ...)   \
 do {                             \
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

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device.
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

int ExecuteInplaceLtTensorOperator(aclrtStream stream) {
  auto ret = 0;
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> otherHostData = {1, 1, 1, 1, 0, 0, 0, 0};

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_INT32, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceLtTensorGetWorkspaceSize(self, other, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnInplaceLtTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceLtTensor failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(selfShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(resultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %lf\n", i, resultData[i]);
  }

  aclDestroyTensor(self);
  aclDestroyTensor(other);

  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Execute the InplaceLtTensor operation.
  ret = ExecuteInplaceLtTensorOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteInplaceLtTensorOperator failed. ERROR: %d\n", ret); return ret);

  // Reset the device and deinitialize ACL.
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
