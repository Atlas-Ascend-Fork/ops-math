# aclnnTriangularSolve

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:31:55.853Z -->

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

- Description: Solves a system of equations with a square upper or lower triangular invertible matrix A and multiple right-hand sides b.
- Formula:

  $$
  AX = b
  $$

  Where $A$ is an upper triangular square matrix (or a lower triangular square matrix when upper is false), with no zero elements on its main diagonal. $b$ and $A$ are two-dimensional matrices or batches of two-dimensional matrices. When the input is a batch, the returned output $X$ is also the corresponding batch. When the main diagonal of $A$ contains zeros, or elements very close to zero, and unitriangular is false, the output may contain $NaN$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnTriangularSolveGetWorkspaceSize` API first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then, call the `aclnnTriangularSolve` API to perform the computation.

```Cpp
aclnnStatus aclnnTriangularSolveGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* A,
  bool             upper,
  bool             transpose,
  bool             unitriangular,
  aclTensor*       xOut,
  aclTensor*       mOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTriangularSolve(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnTriangularSolveGetWorkspaceSize

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
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>b in the formula, the right-hand side of the equation.</td>
      <td><ul><li>The data type is consistent with A.</li><li>self[-2]=A[-2].</li><li>Except for the last two dimensions, the remaining dimensions of A and self satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> relationship.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>A (aclTensor*)</td>
      <td>Input</td>
      <td>A in the formula, the coefficient matrix.</td>
      <td><ul><li>The data type is consistent with that of self.</li><li>The last two axes are equal.</li><li>Except for the last two dimensions, the remaining dimensions of A and self satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> relationship.</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>upper (bool)</td>
      <td>Input</td>
      <td>Computation attribute that controls whether A in the formula participates in the computation as an upper triangular or lower triangular matrix.</td>
      <td>The default is true, indicating that A is an upper triangular square matrix. If upper is false, A is a lower triangular square matrix.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose (bool)</td>
      <td>Input</td>
      <td>Computation attribute that controls whether A or A<sup>T</sup> in the formula participates in the calculation.</td>
      <td>Default is false. When transpose is true, A<sup>T</sup>X=b is computed.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unitriangular (bool)</td>
      <td>Input</td>
      <td>Computation attribute that controls whether A in the formula is treated as a unit triangular matrix.</td>
      <td><ul><li>Default is false.</li><li>When unitriangular is true, the elements on the main diagonal of A are treated as 1, instead of being referenced from A.</li><li>When unitriangular is true, only the FLOAT data type is supported for inputs self and A and outputs xOut and mOut.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>X in the formula, the solution of the equation.</td>
      <td><ul><li>The data type is consistent with self.</li><li>The shape must satisfy the AX=b constraint with A and b after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</li><li>After A and self satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> relationship, the last axis is dim=self[-1].</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mOut (aclTensor*)</td>
      <td>Output</td>
      <td>Copy of the upper (lower) triangle of A after broadcast.</td>
      <td><ul><li>The data type is consistent with self.</li><li>Dimensions of A and self after satisfying the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> relationship. The last axis is dim=A[-1].</li></ul></td>
      <td>FLOAT, DOUBLE, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>-</td>
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
      <td>Returns the op executor containing the operator calculation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

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
      <td>A null pointer exists in the input self, A, xOut, or mOut.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and data formats of self, A, xOut, and mOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self, A, xOut, and mOut do not satisfy the constraints.</td>
    </tr>
  </tbody></table>

## aclnnTriangularSolve

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
      <td>Address of the workspace memory requested on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first-phase API <code>aclnnTriangularSolveGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description: `aclnnTriangularSolve` defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triangular_solve.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to request memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {3, 1};
  std::vector<int64_t> aShape = {3, 3};
  std::vector<int64_t> xOutShape = {3, 1};
  std::vector<int64_t> mOutShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* aDeviceAddr = nullptr;
  void* xOutDeviceAddr = nullptr;
  void* mOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* A = nullptr;
  aclTensor* xOut = nullptr;
  aclTensor* mOut = nullptr;
  bool upper = true;
  bool transpose = false;
  bool unitriangular = false;
  std::vector<float> selfHostData = {1, 2, 3};
  std::vector<float> aHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  std::vector<float> xOutHostData = {-0.2500, -0.1250, 0.5000};
  std::vector<float> mOutHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the A aclTensor.
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT, &A);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the xOut aclTensor.
  ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT, &xOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mOut aclTensor.
  ret = CreateAclTensor(mOutHostData, mOutShape, &mOutDeviceAddr, aclDataType::ACL_FLOAT, &mOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnTriangularSolve.
  ret = aclnnTriangularSolveGetWorkspaceSize(self, A, upper, transpose, unitriangular, xOut, mOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTriangularSolve.
  ret = aclnnTriangularSolve(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolve failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values. Copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto xSize = GetShapeSize(xOutShape);
  std::vector<float> resultData(xSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xOutDeviceAddr,
                    xSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto mSize = GetShapeSize(mOutShape);
  std::vector<float> mResultData(mSize, 0);
  ret = aclrtMemcpy(mResultData.data(), mResultData.size() * sizeof(mResultData[0]), mOutDeviceAddr,
                    mSize * sizeof(mResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < mSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, mResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(A);
  aclDestroyTensor(xOut);
  aclDestroyTensor(mOut);

  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(aDeviceAddr);
  aclrtFree(xOutDeviceAddr);
  aclrtFree(mOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
