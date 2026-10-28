# aclnnSinkhorn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:14:07.304Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Operator function:

  Computes the Sinkhorn distance, which can be used for expert routing in MoE models.

- Formula:

  $$
  p=Sinkhorn(cost, tol)
  $$

  **Input**:
  cost(R, C): 2D cost matrix
  tol: tolerance

  **Initialization**:

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1\_old= d1 \\
  $$

  **Repeat**:

  $$
  d0 = \frac{1}{R * (sum(d1 * cost, 1) + eps)} \\
  d1 = \frac{1}{C * (sum(d0.unsqueeze(1) * cost, 0) + eps)} \\
  error = mean(abs(d1\_old - d1)) \\
  d1\_old = d1
  $$

  Until:

  $$
  error <= tol
  $$

  **Output**:
  $
  p = d1 * cost * d0.unsqueeze(1)
  $

## Function Prototype

Each operator is divided into the [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSinkhornGetWorkspaceSize" API to obtain input parameters and calculate the required workspace size based on the computing process, and then call the "aclnnSinkhorn" API to perform the computation.

- `aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, const aclScalar *tol, aclTensor *p, uint64_t *workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnSinkhornGetWorkspaceSize

- **Parameters**:

    - cost (aclTensor*, computation input): Represents the cost tensor, which is `cost` in the formula. It is an aclTensor on the device. The supported data types are BFLOAT16, FLOAT16, and FLOAT. The value range is [0,1], and normalization can be performed. [Data format](../../../docs/en/context/data_format.md) supports ND. The input is a 2D matrix with no more than 10000 rows and no more than 1024 columns. Supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md).
    - tol (aclScalar*, input): Represents the error for Sinkhorn computation. The supported data type is FLOAT. If a null pointer is passed, tol defaults to 0.0001.
    - p (aclTensor*, computation output): Represents the optimal transport tensor, which is `p` in the formula. It is an aclTensor on the device. The supported data types are BFLOAT16, FLOAT16, and FLOAT. [Data format](../../../docs/en/context/data_format.md) supports ND. The shape dimension is 2. Does not support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md). The data type and shape are the same as those of the input `cost`.
    - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
    - executor (aclOpExecutor**, output parameter): Returns the op executor, which includes computation.

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 301px">
  <col style="width: 137px">
  <col style="width: 714px">
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
      <td>The input cost or p is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of cost and p are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of cost and p cannot be deduced.</td>
    </tr>
  </tbody>
  </table>

## aclnnSinkhorn

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnSinkhornGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnSinkhorn defaults to a deterministic implementation.

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sinkhorn.h"

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

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate, acl initialization
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Device/stream initialization. For details, see the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Perform custom construction based on the API.
  std::vector<int64_t> costShape = {3, 2};
  std::vector<int64_t> pShape = {3, 2};
  void* costDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  aclTensor* cost = nullptr;
  aclScalar* tol = nullptr;
  aclTensor* p = nullptr;
  std::vector<float> costHostData = {0.45f, 0.48f, 0.65f, 0.68f, 0.68f, 0.10f};
  std::vector<float> pHostData(6, 0);

  float tolValue = 0.0001;

  // Create the cost aclTensor.
  ret = CreateAclTensor(costHostData, costShape, &costDeviceAddr, aclDataType::ACL_FLOAT, &cost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the p aclTensor.
  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the tol aclScalar.
  tol = aclCreateScalar(&tolValue, aclDataType::ACL_FLOAT);
  CHECK_RET(tol != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be changed to the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnSinkhorn first-phase API.
  ret = aclnnSinkhornGetWorkspaceSize(cost, tol, p, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnSinkhorn second-phase API.
  ret = aclnnSinkhorn(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhorn failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host side. Modify it based on the API definition.
  auto size = GetShapeSize(pShape);
  std::vector<float> pData(size, 0);
  ret = aclrtMemcpy(pData.data(), pData.size() * sizeof(pData[0]), pDeviceAddr,
                    size * sizeof(pData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("p result[%ld] is: %e\n", i, pData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the API definition.
  aclDestroyTensor(cost);
  aclDestroyTensor(p);
  aclDestroyScalar(tol);

  // 7. Release device resources. Modify it based on the API definition.
  aclrtFree(costDeviceAddr);
  aclrtFree(pDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
