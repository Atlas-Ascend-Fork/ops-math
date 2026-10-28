# aclnnBatchNormStats

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:49:08.925Z -->

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

- Description:
      Computes the mean and the reciprocal of the standard deviation of the single-device input data.

- Formula:

  Mean:

  $$
  \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}
  $$

  Reciprocal of standard deviation:

  $$
  \frac{1}\sigma = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i- \bar{x})^2 + eps}}
  $$

## Function Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBatchNormStatsGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the procedure, and then call the "aclnnBatchNormStats" API to execute the computation.

```Cpp
aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(
  const aclTensor*            input,
  double                      eps,
  aclTensor*                  mean,
  aclTensor*                  invstd,
  uint64_t*                   workspaceSize,
  aclOpExecutor**             executor)
```

```Cpp
aclnnStatus aclnnBatchNormStats(
  void*                       workspace,
  uint64_t                    workspaceSize,
  aclOpExecutor*              executor,
  const aclrtStream           stream)
```

## aclnnBatchNormStatsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
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
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>-</td>
      <td>Supported shapes and data formats:<br>2D (NC).<br>3D (NCL).<br>4D (NCHW).<br>5D (NCDHW).<br>6-8D (ND, where the second dimension is fixed to the channel axis).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Value added to the denominator for numerical stability.</td>
      <td>-</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Output</td>
      <td>Output mean.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd (aclTensor*)</td>
      <td>Output</td>
      <td>Outputs the reciprocal of the standard deviation.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-D</td>
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
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Op executor returned, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b,910" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas training products</term>: the data type of `input` does not support BFLOAT16.
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
      <th>Return value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input, mean, or invstd is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or data format of input, mean, or invstd is not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of input is less than 2 or greater than 8, or the dimension of mean or invstd is not 1.</td>
    </tr>
    <tr>
      <td>The shape of mean or invstd is inconsistent with the channel axis of input.</td>
    </tr>
    <tr>
      <td>The value of the second dimension of input is 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchNormStats

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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnBatchNormStatsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation process.</td>
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

- deterministic computation:
  - aclnnBatchNormStats defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_stats.h"

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
  // Boilerplate: initialize resources.
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
  // Call aclrtMalloc to allocate memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
  // Set deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Custom construction is required based on the API.
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<int64_t> outShape = {3,};
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> meanHostData = {0, 0, 0};
  std::vector<float> invstdHostData = {0, 0, 0};
  double eps = 1e-5;

  // Create the self aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mean aclTensor.
  ret = CreateAclTensor(meanHostData, outShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the invstd aclTensor.
  ret = CreateAclTensor(invstdHostData, outShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnnBatchNormStats API call example
  // 3. Call the CANN operator library API and replace it with the actual API name.
  // Call the first-phase API of aclnnBatchNormStats.
  ret = aclnnBatchNormStatsGetWorkspaceSize(input, eps, mean, invstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBatchNormStats.
  ret = aclnnBatchNormStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormStats failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> meanData(size, 0);
  ret = aclrtMemcpy(meanData.data(), meanData.size() * sizeof(meanData[0]), meanDeviceAddr,
                    size * sizeof(meanData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, meanData[i]);
  }
  std::vector<float> invstdData(size, 0);
  ret = aclrtMemcpy(invstdData.data(), invstdData.size() * sizeof(invstdData[0]), invstdDeviceAddr,
                    size * sizeof(invstdData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("invstd result[%ld] is: %f\n", i, invstdData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);

  // 7. Release device resources. Modify it based on the API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
