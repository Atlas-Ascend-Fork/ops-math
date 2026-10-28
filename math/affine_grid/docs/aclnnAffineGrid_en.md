# aclnnAffineGrid

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:15:39.907Z -->

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

Generates a 2D or 3D grid given a set of 3D affine parameter matrices (theta) and the output image size, where the grid represents the coordinates of points in the original image after affine transformation.

## Function Prototype

Each operator involves a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAffineGridGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the procedure, and then call the "aclnnAffineGrid" API to execute the computation.

```Cpp
aclnnStatus aclnnAffineGridGetWorkspaceSize(
  const aclTensor*   theta,
  const aclIntArray* size,
  bool               alignCorners,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnAffineGrid(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAffineGridGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1545px"><colgroup>
  <col style="width: 248px">
  <col style="width: 128px">
  <col style="width: 307px">
  <col style="width: 289px">
  <col style="width: 131px">
  <col style="width: 121px">
  <col style="width: 175px">
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
      <td>theta(aclTensor*)</td>
      <td>Input</td>
      <td>Affine transformation parameters controlling rotation, scaling, and translation during the affine transformation.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(N, 2, 3) or (N, 3, 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size(aclIntArray*)</td>
      <td>Input</td>
      <td>Output image size.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>(N, C, H, W) or (N, C, D, H, W)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners(bool)</td>
      <td>Input</td>
      <td>Indicates whether corner pixels are aligned.</td>
      <td><ul><li>If True, the corner pixels of the output grid are aligned with the corner pixels of the input grid.</li><li>If False, the center pixels of the output grid are aligned with the center pixels of the input grid.</li><li>Defaults to False.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the coordinates of the affine-transformed image on the original image.</td>
      <td></td>
      <td>Same as theta</td>
      <td>ND</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id7 -->
  <term>Atlas training products</term>: The data type BFLOAT16 is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 302px">
  <col style="width: 135px">
  <col style="width: 710px">
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
      <td>theta, size, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of theta is not supported.</td>
    </tr>
    <tr>
      <td>The data type of out is inconsistent with that of theta.</td>
    </tr>
    <tr>
      <td>The dimension of theta is not 3.</td>
    </tr>
    <tr>
      <td>The length of size is not 4 or 5.</td>
    </tr>
    <tr>
      <td>If the length of size is 4, the shape of theta is not (N, 2, 3); if the length of size is 5, the shape of theta is not (N, 3, 4).</td>
    </tr>
    <tr>
      <td>When the length of size is 4, the shape of out is not (N, H, W, 2); when the length of size is 5, the shape of out is not (N, D, H, W, 3).</td>
    </tr>
  </tbody>
  </table>

## aclnnAffineGrid

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by the first-phase API aclnnAffineGridGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
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
  - aclnnAffineGrid defaults to a deterministic implementation.

- The values of N, H, and W in size must be in the range (0, 100000].

## Example

The following sample code is for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_affine_grid.h"

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the ACL API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> thetaShape = {1, 2, 3};
  std::vector<int64_t> outShape = {1, 2, 3, 2};
  void* thetaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* theta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> thetaHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> sizeData = {1, 1, 2, 3};
  bool alignCorners = false;

  // Create the theta aclTensor.
  ret = CreateAclTensor(thetaHostData, thetaShape, &thetaDeviceAddr, aclDataType::ACL_FLOAT, &theta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the size aclIntArray.
  aclIntArray *size = aclCreateIntArray(sizeData.data(), sizeData.size());
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAffineGrid.
  ret = aclnnAffineGridGetWorkspaceSize(theta, size, alignCorners, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGridGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnAffineGrid.
  ret = aclnnAffineGrid(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAffineGrid failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the definition of the specific API.
  auto length = GetShapeSize(outShape);
  std::vector<float> resultData(length, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    length * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < length; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it based on the definition of the specific API.
  aclDestroyTensor(theta);
  aclDestroyIntArray(size);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify based on the specific API definition.
  aclrtFree(thetaDeviceAddr);
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
