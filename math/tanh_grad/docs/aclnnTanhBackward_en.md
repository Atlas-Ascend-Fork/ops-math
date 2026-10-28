# aclnnTanhBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:26:37.075Z -->

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

- Description: backward propagation of aclnnTanh.

- Formula:

$$
gradInput = gradOutput \cdot (1 -output^2)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the `aclnnTanhBackwardGetWorkspaceSize` API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the `aclnnTanhBackward` API to execute the computation.

```Cpp
aclnnStatus aclnnTanhBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *output,
  aclTensor       *gradInput,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnTanhBackward(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnTanhBackwardGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 203px">
  <col style="width: 124px">
  <col style="width: 212px">
  <col style="width: 309px">
  <col style="width: 305px">
  <col style="width: 114px">
  <col style="width: 135px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>The input <code>gradOutput</code> in the formula.</td>
      <td><ul><li>The data format must be the same as that of output and gradInput.</li><li>The shape must be the same as that of output and gradInput or satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</li><li>The data type must be the same as that of output and gradInput.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output (aclTensor*)</td>
      <td>Input</td>
      <td>The input <code>output</code> in the formula.</td>
      <td><ul><li>Consistent with the data format of gradOutput and gradInput.</li><li>Consistent with the shape of gradOutput and gradInput, or satisfying the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</li><li>Consistent with the data type of gradOutput and gradInput.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td><code>gradInput</code> in the formula.</td>
      <td><ul><li>Consistent with the data format of gradOutput and output.</li><li>Consistent with the shape of gradOutput and output, or the shape after gradOutput and output are broadcast.</li><li>Consistent with the data type of gradOutput and output.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
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
      <td>Returned when gradOutput, output, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or dimension of gradOutput, output, or gradInput is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, output, and gradInput are different.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, output, and gradInput are different.</td>
    </tr>
  </tbody>
  </table>

## aclnnTanhBackward

- **Parameters**

  <table style="table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 158px">
  <col style="width: 140px">
  <col style="width: 845px">
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
      <td>Workspace size requested on the device. It is obtained by the first-phase API aclnnTanhBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnTanhBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_tanh_backward.h"

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
  // Call aclrtMalloc to apply for memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> outputShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* output = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7.0};
  std::vector<float> outputHostData = {1, 1, 1, 2, 2, 2, 3, 3.0};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0.0};
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the output aclTensor.
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library APIs. Replace them with the specific APIs.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTanhBackward.
  ret = aclnnTanhBackwardGetWorkspaceSize(gradOutput, output, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanhBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTanhBackward.
  ret = aclnnTanhBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTanhBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result on the device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(output);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(outputDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
