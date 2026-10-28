# aclnnVarMean

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:56:54.814Z -->

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

- Description: Returns the mean and variance computed from the values of the input tensor along the specified dimension.
- Formula: Assume dim is $i$. The calculation is performed along this dimension. $N$ is the shape of this dimension. Take $self_{i}$ to calculate the mean along this dimension: $meanOut = \bar{self_{i}}$.

  The variance formula is as follows:

  $$
  varOut = \frac{1}{max(0, N - correction)}\sum_{j=0}^{N-1}(self_{ij}-\bar{self_{i}})^2
  $$

  If `keepdim = true`, this dimension is retained after reduction and is set to 1 in the output shape; if `keepdim = false`, this dimension is not retained.
  If dim is nullptr or [], all dimensions are calculated.

## Function Prototype

Each operator involves a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnVarMeanGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnVarMean" to execute the computation.

```Cpp
aclnnStatus aclnnVarMeanGetWorkspaceSize(
  const aclTensor*   self,
  const aclIntArray* dim,
  int64_t            correction,
  bool               keepdim,
  aclTensor*         varOut,
  aclTensor*         meanOut,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnVarMean(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnVarMeanGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 198px">
  <col style="width: 124px">
  <col style="width: 252px">
  <col style="width: 319px">
  <col style="width: 260px">
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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>The <code>self</code> in the formula.</td>
      <td>The shape supports 0 to 8 dimensions.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclIntArray*)</td>
      <td>Input</td>
      <td>The <code>dim</code> in the formula.</td>
      <td>Dimensions for the computation, in the range [-self.dim(), self.dim()-1]. The values must be unique. The supported data type is INT64. If dim is nullptr or [], all dimensions are computed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction (int64_t)</td>
      <td>Input</td>
      <td>The input <code>correction</code> in the formula, indicating the correction value.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim (bool)</td>
      <td>Input</td>
      <td>Whether the dimension along the reduce axis is retained.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut (aclTensor*)</td>
      <td>Output</td>
      <td>Calculation result of the mean.</td>
      <td>Supports 0 to 8 dimensions.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>varOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output <code>varOut</code> in the formula. The computation result of the variance.</td>
      <td>The shape supports 0 to 8 dimensions.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
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
  </tbody>
  </table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The BFLOAT16 data type is not supported.
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
      <td>self, meanOut, or varOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of self, meanOut, or varOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of self exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The value of dim is invalid (data in dim points to the same dimension, or dim is out of the dimension range of self).</td>
    </tr>
    <tr>
      <td>The shapes of self, meanOut, and varOut do not satisfy the deduction rules in the formula.</td>
    </tr>
  </tbody>
  </table>

## aclnnVarMean

- **Parameters:**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnVarMeanGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**
  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnVarMean defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about how to compile and run, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_var_mean.h"

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
  // Boilerplate: resource initialization.
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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2,4};
  std::vector<int64_t> outShape = {2,1};
  void* selfDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* var = nullptr;
  aclTensor* mean = nullptr;
  std::vector<float> selfHostData = {0.0, 1.1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> dimData = {1};
  int64_t correction = 1;
  bool keepdim = true;
  std::vector<float> varHostData = {0.0, 0};
  std::vector<float> meanHostData = {0.0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the dim aclIntArray.
  dim = aclCreateIntArray(dimData.data(), dimData.size());
  CHECK_RET(dim != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(varHostData, outShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(meanHostData, outShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnVarMean.
  ret = aclnnVarMeanGetWorkspaceSize(self, dim, correction, keepdim, var, mean, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnVarMean.
  ret = aclnnVarMean(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnVarMean failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> meanData(size, 0);
  ret = aclrtMemcpy(meanData.data(), meanData.size() * sizeof(meanData[0]), meanDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResult[%ld] is: %f\n", i, meanData[i]);
  }
  std::vector<float> varData(size, 0);
  ret = aclrtMemcpy(varData.data(), varData.size() * sizeof(varData[0]), varDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("varResult[%ld] is: %f\n", i, varData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(dim);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);

  // 7. Release the device resources. Modify it based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(meanDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
