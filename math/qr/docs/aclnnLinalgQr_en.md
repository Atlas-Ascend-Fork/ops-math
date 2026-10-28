# aclnnLinalgQr

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:19:21.161Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Performs orthogonal decomposition on the input tensor.
- Formula:

  $$
  A = QR
  $$

  Where $A$ is the input tensor with at least two dimensions, and $A$ can be expressed as the product of an orthogonal matrix $Q$ and an upper triangular matrix $R$.

- Example:

  ```text
  A = tensor([[1, 2], [3, 4]], dtype=torch.float)
  q,r = linalg_qr(A, mode='reduced')
  q = tensor([[-0.3162, -0.9487],
             [-0.9487, 0.3162]])
  r = tensor([[-3.1623, -4.4272],
             [0.0000, -0.6325]])
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnLinalgQrGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the `aclnnLinalgQr` API to execute the computation.

```Cpp
aclnnStatus aclnnLinalgQrGetWorkspaceSize(
  const aclTensor* self,
  int64_t          mode,
  aclTensor*       Q,
  aclTensor*       R,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnLinalgQr(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnLinalgQrGetWorkspaceSize

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
      <th>Dimension(shape)</th>
      <th>Non-Contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor A in the formula.</td>
      <td>The shape must meet the constraints with Q and R.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode (int64_t)</td>
      <td>Input</td>
      <td>Controls the computation attribute of the output forms of Q and R in the formula.</td>
      <td><ul><li>When mode is 0, the <code>reduced</code> mode is used. For input A(*, m, n), Q(*, m, k) and R(*, k, n) of reduced size are output, where k is the minimum value of m and n.</li><li>When mode is 1, the <code>complete</code> mode is used. For input A(*, m, n), Q(*, m, m) and R(*, m, n) of full size are output.</li><li>When mode is 2, the <code>r</code> mode is used. Only R(*, k, n) in the <code>reduced</code> mode is computed, where k is the minimum value of m and n, and the returned Q is an empty Tensor.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Q (aclTensor*)</td>
      <td>Output</td>
      <td>Q in the formula. Orthogonal matrix output by the orthogonal decomposition.</td>
      <td>The shape is Q(*, m, m), Q(*, m, k), or empty, where k is the minimum value of m and n, and the data format must be the same as those of <code>self</code> and R.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from mode</td>
      <td>√</td>
    </tr>
    <tr>
      <td>R (aclTensor*)</td>
      <td>Output</td>
      <td>R in the formula, the upper triangular matrix output by the orthogonal decomposition.</td>
      <td>The shape is R(*, m, n) or R(*, k, n), where k is the minimum value of m and n, and the data format must be consistent with <code>self</code> and Q.</td>
      <td>FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>Derived from mode</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
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
      <td>Null pointers exist in the input self, Q, or R.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and data formats of self, Q, or R are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self, Q, or R do not meet the constraints.</td>
    </tr>
    <tr>
      <td>mode is not within the optional range.</td>
    </tr>
  </tbody></table>

## aclnnLinalgQr

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first-phase API <code>aclnnLinalgQrGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description: `aclnnLinalgQr` defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linalg_qr.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMemcpy to copy the host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
    std::vector<int64_t>& selfShape, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    void** selfDeviceAddr, void** qOutDeviceAddr, void** rOutDeviceAddr, aclTensor** self, aclTensor** qOut,
    aclTensor** rOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> qOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> rOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(qOutHostData, qOutShape, qOutDeviceAddr, aclDataType::ACL_FLOAT, qOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(rOutHostData, rOutShape, rOutDeviceAddr, aclDataType::ACL_FLOAT, rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpAPI(
    aclTensor* self, aclTensor* qOut, aclTensor* rOut, int64_t mode, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* qOutDeviceAddr, void* rOutDeviceAddr, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnLinalgQrGetWorkspaceSize(self, mode, qOut, rOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnLinalgQr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQr failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy qOut.
  auto size1 = GetShapeSize(qOutShape);
  std::vector<float> resultData1(size1, 0);
  ret = aclrtMemcpy(
      resultData1.data(), resultData1.size() * sizeof(resultData1[0]), qOutDeviceAddr, size1 * sizeof(resultData1[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size1; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
  }

  // Copy rOut.
  auto size2 = GetShapeSize(rOutShape);
  std::vector<float> resultData2(size2, 0);
  ret = aclrtMemcpy(
      resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rOutDeviceAddr, size2 * sizeof(resultData2[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size2; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> qOutShape = {1, 1, 4, 4};
  std::vector<int64_t> rOutShape = {1, 1, 4, 4};

  void* selfDeviceAddr = nullptr;
  void* qOutDeviceAddr = nullptr;
  void* rOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* qOut = nullptr;
  aclTensor* rOut = nullptr;

  ret = CreateInputs(
      selfShape, qOutShape, rOutShape, &selfDeviceAddr, &qOutDeviceAddr, &rOutDeviceAddr, &self, &qOut, &rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t mode = 0;
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpAPI(
      self, qOut, rOut, mode, &workspaceAddr, workspaceSize, qOutDeviceAddr, rOutDeviceAddr, qOutShape, rOutShape,
      stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Destroy the tensor.
  aclDestroyTensor(self);
  aclDestroyTensor(qOut);
  aclDestroyTensor(rOut);

  // Release the device memory.
  aclrtFree(selfDeviceAddr);
  aclrtFree(qOutDeviceAddr);
  aclrtFree(rOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
