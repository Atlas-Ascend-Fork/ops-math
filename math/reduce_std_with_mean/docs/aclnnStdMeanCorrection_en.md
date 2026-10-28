# aclnnStdMeanCorrection

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:49:10.637Z -->

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

- Description: Computes the sample standard deviation and mean.
- Formula:
  Assume dim is $i$, then calculation is performed on this dimension. $N$ is the shape of this dimension. Take $self_{i}$ and calculate the mean on this dimension $\bar{x_{i}}$.

  $$
  \left\{
  \begin{array} {rcl}
  meanOut& &= \bar{x_{i}}\\
  stdOut& &= \sqrt{\frac{1}{max(0, N - \delta N)}\sum_{j=0}^{N-1}(self_{ij}-\bar{x_{i}})^2}
  \end{array}
  \right.
  $$

  When `keepdim = true`, the dimension is retained after reduction, and its value in the output shape is 1. When `keepdim = false`, the dimension is not retained.

## Function Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnStdMeanCorrectionGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the calculation process. Then, call the "aclnnStdMeanCorrection" API to perform the calculation.

```Cpp
aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(
  const aclTensor*   self,
  const aclIntArray* dim,
  int64_t            correction,
  bool               keepdim,
  aclTensor*         stdOut,
  aclTensor*         meanOut,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnStdMeanCorrection(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnStdMeanCorrectionGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 248px">
  <col style="width: 131px">
  <col style="width: 265px">
  <col style="width: 218px">
  <col style="width: 264px">
  <col style="width: 124px">
  <col style="width: 142px">
  <col style="width: 148px">
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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>The self in the formula.</td>
      <td>-</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclIntArray*)</td>
      <td>Input</td>
      <td>The dim in the formula, indicating the dimension involved in the calculation.</td>
      <td>Value range: [-self.dim(), self.dim()-1], and duplicate values are not allowed.<br>When dim is nullptr or [], calculation is performed on all dimensions.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction (int64_t)</td>
      <td>Input</td>
      <td>Value of δN in the formula, the correction value.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim (bool)</td>
      <td>Input</td>
      <td>keepdim in the formula. Specifies whether to retain the dimensions of the input tensor in the output tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stdOut (aclTensor*)</td>
      <td>Output</td>
      <td>stdOut in the formula.</td>
      <td>-</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOut (aclTensor*)</td>
      <td>Output</td>
      <td>meanOut in the formula.</td>
      <td>-</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
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
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 737px">
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
      <td>When self, stdOut, and meanOut are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of self, stdOut, and meanOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>A dimension in the dim array is out of the dimension range of self.</td>
    </tr>
    <tr>
      <td>Elements in the dim array are duplicated.</td>
    </tr>
    <tr>
      <td>An error is reported when the shape of stdOut is as follows:<br>When keepdim is true, stdOut.shape != self.shape (the shape with the specified dimension dim set to 1);<br>When keepdim is false, stdOut.shape != self.shape (the shape with the specified dimension dim removed).</td>
    </tr>
    <tr>
      <td>An error is reported when the shape of meanOut is as follows:<br>When keepdim is true, meanOut.shape != self.shape (the shape with the specified dimension dim set to 1);<br>When keepdim is false, meanOut.shape != self.shape (the shape with the specified dimension dim removed).</td>
    </tr>
  </tbody>
  </table>

## aclnnStdMeanCorrection

- **Parameters**

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
      <td>Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnStdMeanCorrectionGetWorkspaceSize.</td>
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
  - aclnnStdMeanCorrection defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_std_mean_correction.h"

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
  // Call aclrtMalloc to apply for device-side memory.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> stdOutShape = {2, 4};
  std::vector<int64_t> meanOutShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* stdOutDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* stdOut = nullptr;
  aclTensor* meanOut = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24};
  std::vector<float> stdOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<float> meanOutHostData = {1, 2, 3, 4, 5, 6, 7, 8.0};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = false;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the stdOut aclTensor.
  ret = CreateAclTensor(stdOutHostData, stdOutShape, &stdOutDeviceAddr, aclDataType::ACL_FLOAT, &stdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const aclIntArray *dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Change it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnStdMeanCorrection.
  ret = aclnnStdMeanCorrectionGetWorkspaceSize(self, dim, correction, keepdim, stdOut, meanOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrectionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnStdMeanCorrection.
  ret = aclnnStdMeanCorrection(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStdMeanCorrection failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(stdOutShape);
  std::vector<float> stdResultData(size, 0);
  ret = aclrtMemcpy(stdResultData.data(), stdResultData.size() * sizeof(stdResultData[0]), stdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("stdResultData[%ld] is: %f\n", i, stdResultData[i]);
  }

  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(stdOut);
  aclDestroyTensor(meanOut);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(stdOutDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
