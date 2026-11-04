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

- Interface function:

  Computes the Sinkhorn optimal transport plan, which can be used for expert routing in MoE models.

- Formula:

  $$
  p=Sinkhorn(cost, tol)
  $$

  **Input**:

  - cost(R, C): 2D cost matrix
  - tol: convergence tolerance

  **Initialization**:

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1_old = d1
  $$

  **Repeat**:

  $$
  d0 = \frac{1}{R * (sum(d1 * cost, 1) + eps)} \\
  d1 = \frac{1}{C * (sum(d0.unsqueeze(1) * cost, 0) + eps)} \\
  error = mean(abs(d1_old - d1)) \\
  d1_old = d1
  $$

  Until the following condition is satisfied:

  $$
  error <= tol
  $$

  **Output**:

  $$
  p = d1 * cost * d0.unsqueeze(1)
  $$

## Function Prototype

Each operator is called through a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSinkhornGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnSinkhorn" API to execute the computation.

```cpp
aclnnStatus aclnnSinkhornGetWorkspaceSize(
  const aclTensor  *cost,
  const aclScalar  *tol,
  aclTensor        *p,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSinkhorn(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnSinkhornGetWorkspaceSize

- **Parameters**:

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cost (const aclTensor*)</td>
      <td>Input</td>
      <td>The cost tensor, i.e., <code>cost</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The recommended value range is [0,1]; values outside this range are still computed and may produce Inf or NaN.</li><li>2D matrix with no fixed row limit, up to 4096 columns, and up to UINT32_MAX total elements.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tol (const aclScalar*)</td>
      <td>Optional input</td>
      <td>The convergence tolerance for Sinkhorn computation, i.e., <code>tol</code> in the formula.</td>
      <td><ul><li>If a null pointer is passed, tol defaults to 0.0001.</li><li>All FLOAT values are supported; NaN, ±Inf, and negative values participate in the termination determination according to floating-point comparison semantics.</li></ul></td>
      <td>Scalar types convertible to FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p (aclTensor*)</td>
      <td>Output</td>
      <td>The optimal transport tensor, i.e., <code>p</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Shape must be the same as cost.</li><li>When cost is FLOAT, p must be FLOAT and contiguous; when cost is FLOAT16/BFLOAT16, p supports FLOAT, FLOAT16, or BFLOAT16.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√<sup>1</sup></td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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
  </tbody></table>

  > <sup>1</sup> Non-contiguous p is not supported when cost is a non-empty FLOAT tensor; non-contiguous p is supported when cost is FLOAT16/BFLOAT16 or when the tensor is empty.

- **Return Value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and reports an error in the following scenarios:

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
      <td>cost, p, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of cost and p are not within the supported range.</td>
    </tr>
    <tr>
      <td>cost is FLOAT, but p is not FLOAT.</td>
    </tr>
    <tr>
      <td>The shape of cost is not 2-dimensional, a dimension of cost is negative, the number of columns exceeds 4096, or the total number of elements exceeds UINT32_MAX.</td>
    </tr>
    <tr>
      <td>The shapes of cost and p are inconsistent.</td>
    </tr>
    <tr>
      <td>cost is non-empty and FLOAT, and p is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td>The data format of cost or p is not ND.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561000</td>
      <td>An internal error occurred. Contact technical support with the complete logs.</td>
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

- **Return Value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnSinkhorn uses a deterministic implementation by default.

- Convergence guard: at most 10000 iterations are performed internally. If the error has not converged to `tol` within 10000 iterations, the computation stops and the state of the 10000th iteration is returned as the output.

- Special value termination: `!(error > tol)` is strictly used; NaN comparison terminates the iteration, while `+Inf` compared with a finite tol continues the iteration and is ultimately still protected by the 10000-iteration upper limit.

- Empty tensor: when R or C is 0, the call succeeds with workspaceSize of 0.

- Numerical range: `[0,1]` is the recommended domain only. Values outside this range are still computed and may produce Inf or NaN.

## Example

The sample code is as follows, for reference only. For the detailed compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                  \
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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                             shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Device/stream initialization
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and output
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

  ret = CreateAclTensor(costHostData, costShape, &costDeviceAddr, aclDataType::ACL_FLOAT, &cost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  tol = aclCreateScalar(&tolValue, aclDataType::ACL_FLOAT);
  CHECK_RET(tol != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return -1);

  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSinkhornGetWorkspaceSize(cost, tol, p, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSinkhorn(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhorn failed. ERROR: %d\n", ret); return ret);

  // 4. Synchronously wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value
  auto size = GetShapeSize(pShape);
  std::vector<float> pData(size, 0);
  ret = aclrtMemcpy(pData.data(), pData.size() * sizeof(pData[0]), pDeviceAddr,
                    size * sizeof(pData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("p result[%ld] is: %e\n", i, pData[i]);
  }

  // 6. Release the resources
  aclDestroyTensor(cost);
  aclDestroyTensor(p);
  aclDestroyScalar(tol);

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
