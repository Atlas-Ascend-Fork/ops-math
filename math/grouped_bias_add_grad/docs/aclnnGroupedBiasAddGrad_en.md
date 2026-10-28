# aclnnGroupedBiasAddGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:14:06.580Z -->

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

- Description: Backward computation of GroupedBiasAdd. The extended API of this API is [aclnnGroupedBiasAddGradV2](./aclnnGroupedBiasAddGradV2_en.md).
- Formula:<br>
  (1) When the optional input groupIdxOptional is provided:

  $$
  out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
  $$

  Where gradY is a 2D tensor, H indicates the dimension size of the last dimension of gradY, and G indicates the dimension size of dimension 0 of groupIdxOptional, that is, groupIdxOptional has G numbers. groupIdxOptional(j) denotes the size of the j-th number. After computation, out is a 2D tensor with shape (G, H).<br>
  (2) When the optional input groupIdxOptional is not provided:

  $$
  out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
  $$

  Where gradY is a 3D tensor, G, C, and H indicate the dimension sizes of dimensions 0-2 of gradY, respectively. After computation, out is a 2D tensor with shape (G, H).

- Example:<br>
(1) When the optional input groupIdxOptional is provided:<br>
  The shape of gradY is (1000, 30), and groupIdxOptional is (400, 600, 1000). gradY is divided into 3 groups, and the number of rows accumulated in each group is 400, 200, and 400 in sequence. After computation, the shape of out is (3, 30).<br>
(2) When the optional input groupIdxOptional is not provided:<br>
  The shape of gradY is (10, 100, 30). gradY is divided into 10 groups, and the number of rows accumulated in each group is 100. After computation, the shape of out is (10, 30).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedBiasAddGradGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnGroupedBiasAddGrad" API to perform computation.

```Cpp
aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(
  const aclTensor*  gradY,
  const aclTensor*  groupIdxOptional,
  aclTensor*        out,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnGroupedBiasAddGrad(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGroupedBiasAddGradGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradY (aclTensor*)</td>
      <td>Input</td>
      <td>Backward propagation gradient, which is gradY in the formula.</td>
      <td>When the optional input groupIdxOptional is provided, the shape only supports 2 dimensions.<br>When the optional input groupIdxOptional is not provided, the shape only supports 3 dimensions.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIdxOptional (aclTensor*)</td>
      <td>Input</td>
      <td>End position of each group, which is groupIdxOptional in the formula.</td>
      <td>Only 1D is supported.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of bias, the out in the formula.</td>
      <td>The data type must be the same as that of gradY. Only 2D shapes are supported.<br> If the input is 3D (G, C, H), the output shape is (G, H);<br>If the input is 2D (GB, H) and group_idx is (G), the output shape is (G, H).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be side-applied on the Device.</td>
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
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>When gradY or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type/dimension of gradY or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The dimension relationships of gradY, groupIdxOptional, and out do not match.</td>
    </tr>
    <tr>
      <td>The number of groups exceeds 2048.</td>
    </tr>
  </tbody></table>

## aclnnGroupedBiasAddGrad

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
      <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnGroupedBiasAddGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedBiasAddGrad defaults to a deterministic implementation.
- groupIdxOptional supports a maximum of 2048 elements.
- When the optional input groupIdxOptional is provided, ensure that the Tensor data is in ascending order and the last value equals the dimension size of dim 0 in gradY.
- When the optional input groupIdxOptional is present, the tensor values must not exceed the INT32 maximum value and must be non-negative.

## Example

The sample code is as follows for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_bias_add_grad.h"

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

  // Call the aclCreateTensor API to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> gradYShape = {40, 10};
  std::vector<int64_t> groupIdxShape = {4};
  std::vector<int64_t> outShape = {4, 10};
  void* gradYDeviceAddr = nullptr;
  void* groupIdxDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradY = nullptr;
  aclTensor* groupIdx = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradYHostData(400, 1.0);

  std::vector<int32_t> groupIdxHostData = {5, 15, 30, 40};
  std::vector<float> outHostData(40, 0.0);

  // Create gradY aclTensor.
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create groupIdxOptional aclTensor.
  ret = CreateAclTensor(groupIdxHostData, groupIdxShape, &groupIdxDeviceAddr, aclDataType::ACL_INT32, &groupIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupedBiasAddGrad.
  ret = aclnnGroupedBiasAddGradGetWorkspaceSize(gradY, groupIdx, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGroupedBiasAddGrad.
  ret = aclnnGroupedBiasAddGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGrad failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device-side memory to the host side. Modify it based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify it based on the definition of the specific API.
  aclDestroyTensor(gradY);
  aclDestroyTensor(groupIdx);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(groupIdxDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(gradYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
