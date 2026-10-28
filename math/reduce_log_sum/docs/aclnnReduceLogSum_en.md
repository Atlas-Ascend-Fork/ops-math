# aclnnReduceLogSum

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:32:16.311Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Returns the logarithm of the sum of each row in the input tensor along the given dimension.

## Function Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnReduceLogSumGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator. Then, call the "aclnnReduceLogSum" API to perform computation.

```Cpp
aclnnStatus aclnnReduceLogSumGetWorkspaceSize(
  const aclTensor*   data,
  const aclIntArray* axes,
  bool               keepDims,
  bool               noopWithEmptyAxes,
  aclTensor*         reduce,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnReduceLogSum(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnReduceLogSumGetWorkspaceSize

- **Parameters**:

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 210px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 140px">
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
      <th>Dimensions(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>data(aclTensor*)</td>
      <td>Input</td>
      <td>Represents the tensor for computation.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axes (aclIntArray*)</td>
      <td>Input</td>
      <td>Specifies the dimensions for computation.</td>
      <td>Value range: [-data.dim(), data.dim()-1].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepDims (bool)</td>
      <td>Input</td>
      <td>Specifies whether to keep the dimensions of the input tensor in the output tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>noopWithEmptyAxes (bool)</td>
      <td>Input</td>
      <td>Specifies the behavior when axes is empty.</td>
      <td><ul><li>false: computes over all axes;</li><li>true: no computation is performed, and the output tensor equals the input tensor.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduce (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the computed result.</td>
      <td>Empty tensor is supported.<br>The data type must be the same as that of data.<br>The shape must be the shape of data after computation.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b,310p" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `data` and `reduce` do not support BFLOAT16.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1146px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 735px">
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
      <td>Input data, axes, or reduce is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of data or reduce is not supported.</td>
    </tr>
    <tr>
      <td>The reduce shape does not match the actual shape.</td>
    </tr>
    <tr>
      <td>A dimension in the axes array is out of the dimension range of the input tensor.</td>
    </tr>
    <tr>
      <td>Axes in the axes array are duplicated.</td>
    </tr>
  </tbody>
  </table>

## aclnnReduceLogSum

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
      <td>Workspace memory address on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size on the Device, obtained by the first-phase API aclnnReduceLogSumGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnReduceLogSum defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_reduce_log_sum.h"

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
  // 1. (Boilerplate) Initialize device/context/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> dataShape = {4, 2};
  std::vector<int64_t> outShape = {2};
  void* dataDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* data = nullptr;
  aclIntArray* axes = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> dataHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData = {0, 0};
  std::vector<int64_t> axesData = {0};
  bool keepDims = false;
  bool noopWithEmptyAxes = false;
  // Create the data aclTensor.
  ret = CreateAclTensor(dataHostData, dataShape, &dataDeviceAddr, aclDataType::ACL_FLOAT, &data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the axes aclIntArray.
  axes = aclCreateIntArray(axesData.data(), 1);
  CHECK_RET(axes != nullptr, return ret);
  // 3. Call the CANN operator library API. Replace the API name with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API aclnnReduceLogSum.
  ret = aclnnReduceLogSumGetWorkspaceSize(data, axes, keepDims, noopWithEmptyAxes, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnReduceLogSum.
  ret = aclnnReduceLogSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReduceLogSum failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify it according to the actual API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclIntArray. Modify it according to the actual API definition.
  aclDestroyTensor(data);
  aclDestroyIntArray(axes);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the actual API definition.
  aclrtFree(dataDeviceAddr);
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
