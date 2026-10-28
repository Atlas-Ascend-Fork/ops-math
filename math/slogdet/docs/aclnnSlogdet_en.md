# aclnnSlogdet

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:13:49.313Z -->

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

- Description: Computes the sign and natural logarithm of the determinant of input self.
- Formula:

  $$
  signOut = sign(det(self))     \\
  logOut = log(abs(det(self)))
  $$

  where det indicates the determinant calculation, and abs indicates the absolute value calculation. If the result of $det(self)$ is 0, then $logOut = -inf$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnSlogdetGetWorkspaceSize` API first to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the `aclnnSlogdet` API to execute the computation.

```Cpp
aclnnStatus aclnnSlogdetGetWorkspaceSize(
  const aclTensor* self,
  aclTensor*       signOut,
  aclTensor*       logOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSlogdet(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnSlogdetGetWorkspaceSize

- **Parameters**

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
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>The input matrix, corresponding to <code>self</code> in the formula.</td>
      <td>Shape in the (*, n, n) format, where <code>*</code> denotes zero or more batch dimensions and n is any positive integer.</td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2 or higher</td>
      <td>√</td>
    </tr>
    <tr>
      <td>signOut (aclTensor*)</td>
      <td>Output</td>
      <td>The sign of the determinant, corresponding to <code>signOut</code> in the formula.</td>
      <td><ul><li>Must satisfy the deduction relationship with <code>self</code>.</li><li>If <code>self</code> is of COMPLEX type, <code>signOut</code> cannot be of non-COMPLEX type.</li><li>The shape is consistent with the batch of <code>self</code>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>consistent with the batch of self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>logOut (aclTensor*)</td>
      <td>Output</td>
      <td><code>logOut</code> in the formula, which is the natural logarithm of the determinant.</td>
      <td><ul><li>Must satisfy the deduction relationship with <code>self</code>.</li><li>If <code>self</code> is of COMPLEX type, <code>logOut</code> cannot be of non-COMPLEX type.</li><li>The shape is consistent with the batch of <code>self</code>.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Consistent with the batch of self.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A null pointer exists in self, signOut, or logOut.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and data formats of self, signOut, and logOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of self does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The shapes of signOut and logOut do not satisfy the constraints.</td>
    </tr>
  </tbody></table>

## aclnnSlogdet

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first-phase API <code>aclnnSlogdetGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: `aclnnSlogdet` defaults to a deterministic implementation.
- The input data does not support overflow values `Inf`/`NaN`.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_slogdet.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
      *deviceAddr);
  return 0;
}

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& signOutShape, std::vector<int64_t>& logOutShape,
    void** selfDeviceAddr, void** signOutDeviceAddr, void** logOutDeviceAddr, aclTensor** self, aclTensor** signOut,
    aclTensor** logOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<float> signOutHostData = {0, 0, 0};
  std::vector<float> logOutHostData = {0, 0, 0};

  // Create the self aclTensor.
  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the signOut aclTensor.
  ret = CreateAclTensor(signOutHostData, signOutShape, signOutDeviceAddr, aclDataType::ACL_FLOAT, signOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the logOut aclTensor.
  ret = CreateAclTensor(logOutHostData, logOutShape, logOutDeviceAddr, aclDataType::ACL_FLOAT, logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* signOut, aclTensor* logOut, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* signOutDeviceAddr, void* logOutDeviceAddr, std::vector<int64_t>& signOutShape,
    std::vector<int64_t>& logOutShape, aclrtStream stream)
{
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSlogdet.
  auto ret = aclnnSlogdetGetWorkspaceSize(self, signOut, logOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdetGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  // Call the second-phase API of aclnnSlogdet.
  ret = aclnnSlogdet(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlogdet failed. ERROR: %d\n", ret); return ret);

  // Synchronize.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy signOut.
  auto sizeSign = GetShapeSize(signOutShape);
  std::vector<float> resultData(sizeSign, 0);
  ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), signOutDeviceAddr, sizeSign * sizeof(resultData[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeSign; i++) {
    LOG_PRINT("signout result[%ld] is: %f\n", i, resultData[i]);
  }

  // Copy logOut.
  auto sizeLog = GetShapeSize(logOutShape);
  std::vector<float> logResultData(sizeLog, 0);
  ret = aclrtMemcpy(
      logResultData.data(), logResultData.size() * sizeof(logResultData[0]), logOutDeviceAddr,
      sizeLog * sizeof(logResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < sizeLog; i++) {
    LOG_PRINT("logout result[%ld] is: %f\n", i, logResultData[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  // 1. Initialize device/stream.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 2. Construct inputs and outputs.
  std::vector<int64_t> selfShape = {3, 2, 2};
  std::vector<int64_t> signOutShape = {3};
  std::vector<int64_t> logOutShape = {3};

  void* selfDeviceAddr = nullptr;
  void* signOutDeviceAddr = nullptr;
  void* logOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* signOut = nullptr;
  aclTensor* logOut = nullptr;

  ret = CreateInputs(
      selfShape, signOutShape, logOutShape, &selfDeviceAddr, &signOutDeviceAddr, &logOutDeviceAddr, &self, &signOut,
      &logOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator API.
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      self, signOut, logOut, &workspaceAddr, workspaceSize, signOutDeviceAddr, logOutDeviceAddr, signOutShape,
      logOutShape, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 6. Release aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(signOut);
  aclDestroyTensor(logOut);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(signOutDeviceAddr);
  aclrtFree(logOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
