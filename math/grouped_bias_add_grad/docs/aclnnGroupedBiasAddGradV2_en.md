# aclnnGroupedBiasAddGradV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:14:38.450Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: not supported
<!-- end id6 -->

## Function

- Description: Performs backward computation of groupBiasAdd. This API extends the functionality of the [aclnnGroupedBiasAddGrad](./aclnnGroupedBiasAddGrad_en.md) API by adding the groupIdxType attribute, which specifies the type of groupIdx.
- Formula:<br>
(1) When the optional input groupIdxOptional is provided and groupIdxType is 0:

$$
out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
$$

&emsp;&emsp;(2) When the optional input groupIdxOptional is provided and groupIdxType is 1:

$$
groupIdx(j) = \sum_{i=0}^{j} groupIdxOptional(i), j=0,...,G-1
$$

$$
out(G,H) = \begin {cases} \sum_{i=groupIdx(j-1)}^{groupIdx(j)} gradY(i,H), & 1 \leq j \leq G-1 \\ \sum_{i=0}^{groupIdx(j)} gradY(i, H), & j=0 \end {cases}
$$

&emsp;&emsp;Here, gradY has 2 dimensions, H indicates the dimension size of the last dimension of gradY, G indicates the dimension size of dimension 0 of groupIdxOptional, that is, groupIdxOptional has G numbers, groupIdxOptional(j) indicates the size of the j-th number, and after computation, out has 2 dimensions with shape (G, H).<br>
&emsp;&emsp;(3) When no optional input groupIdxOptional is provided:

$$
out(G, H) = \sum_{i=0}^{C} gradY(G, i, H)
$$

&emsp;&emsp;Here, gradY has 3 dimensions, with G, C, and H representing the sizes of the 0th to 2nd dimensions of gradY respectively. After computation, out is 2-dimensional with shape (G, H).

- Example:<br>
(1) When the optional input groupIdxOptional is provided and groupIdxType is 0:<br>
  The shape of gradY is (1000, 30), and groupIdxOptional is (400, 600, 1000). gradY is divided into 3 groups, and the number of rows for per-group accumulation is 400, 200, and 400 in turn. After computation, the shape of out is (3, 30).<br>
(2) When the optional input groupIdxOptional is provided and groupIdxType is 1:<br>
  The shape of gradY is (1000, 30), and groupIdxOptional is (400, 210, 390). gradY is divided into 3 groups, and the number of rows for per-group accumulation is 400, 210, and 390 in turn. After computation, the shape of out is (3, 30).<br>
(3) When no optional input groupIdxOptional is provided:<br>
  The shape of gradY is (10, 100, 30). gradY is divided into 10 groups, and the number of rows for per-group accumulation is 100 in each group. After computation, the shape of out is (10, 30).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedBiasAddGradV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the computation flow of the operator, and then call the "aclnnGroupedBiasAddGradV2" API to perform computation.

```c++
aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(
  const aclTensor *gradY,
  const aclTensor *groupIdxOptional,
  int64_t          groupIdxType,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnGroupedBiasAddGradV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedBiasAddGradV2GetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1692px"><colgroup>
  <col style="width: 232px">
  <col style="width: 126px">
  <col style="width: 294px">
  <col style="width: 294px">
  <col style="width: 197px">
  <col style="width: 197px">
  <col style="width: 197px">
  <col style="width: 155px">
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
      <td>gradY</td>
      <td>Input</td>
      <td>Backward propagation gradient, gradY in the formula.</td>
      <td>When the optional input groupIdxOptional is provided, only 2-dimension shapes are supported. When no optional input groupIdxOptional is provided, only 3-dimension shapes are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIdxOptional</td>
      <td>Input</td>
      <td>End position of each group, groupIdxOptional in the formula.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIdxType</td>
      <td>groupIdxType</td>
      <td>Indicates the type of groupIdx.</td>
      <td>0: indicates that the value in groupIdxOptional is the end index of each group.<br>1: indicates that the value in groupIdxOptional is the size of each group.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient of bias, which is out in the formula.</td>
      <td>-</td>
      <td>Same as gradY.</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes computation flow of the operator.</td>
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

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 286px">
  <col style="width: 124px">
  <col style="width: 739px">
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
      <td>gradY or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types or dimensions of gradY, groupIdxOptional, and out are not supported.</td>
    </tr>
    <tr>
      <td>The dimension relationships of gradY, groupIdxOptional, and out are mismatched.</td>
    </tr>
    <tr>
      <td>The number of groups exceeds 2048.</td>
    </tr>
    <tr>
      <td>The value of groupIdxType is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedBiasAddGradV2

- **Parameters:**

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
      <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by calling the first-phase API aclnnGroupedBiasAddGradV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which includes computation of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnGroupedBiasAddGradV2 defaults to a deterministic implementation.
- groupIdxOptional supports a maximum of 2048 elements.
- When the optional input groupIdxOptional is present, the tensor values must not exceed the INT32 maximum value and must be non-negative.
- When the optional input groupIdxOptional is present and groupIdxType is 0, the tensor data must be in ascending order, and the last value must equal the size of the 0th dimension of gradY.
- With optional input groupIdxOptional and groupIdxType set to 1, the sum of tensor values must equal the dimension size of the 0th dimension of gradY.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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

  // Call aclrtMemcpy to copy data from the host side to the device-side memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Custom construction is required based on the API.
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
  std::vector<int32_t> groupIdxHostData = {5, 15, 10, 10};
  std::vector<float> outHostData(40, 0.0);
  int64_t groupIdxType = 1;

  // Create the gradY aclTensor.
  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the groupIdxOptional aclTensor.
  ret = CreateAclTensor(groupIdxHostData, groupIdxShape, &groupIdxDeviceAddr, aclDataType::ACL_INT32, &groupIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupedBiasAddGradV2.
  ret = aclnnGroupedBiasAddGradV2GetWorkspaceSize(gradY, groupIdx, groupIdxType, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupedBiasAddGradV2.
  ret = aclnnGroupedBiasAddGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedBiasAddGradV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result on the device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(gradY);
  aclDestroyTensor(groupIdx);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
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
