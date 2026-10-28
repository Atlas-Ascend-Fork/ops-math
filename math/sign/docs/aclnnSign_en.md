# aclnnSign

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:04:19.598Z -->

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

- Operator function: Performs the Sign function element-wise on the input tensor and outputs the result tensor.

- Formula:

  $$
  resultput_i = \left\{
  \begin{aligned}
  1,\quad input_i > 0\\
  0,\quad input_i = 0\\
  -1,\quad input_i < 0
  \end{aligned}
  \right.
  $$

- Formula (BOOL type case):

  $$
  resultput_i = \left\{
  \begin{aligned}
  \text{True},\quad input_i = \text{True}\\
  \text{False},\quad input_i = \text{False}\\
  \end{aligned}
  \right.
  $$

- Formula (complex number case, where real and imag denote the real part and imaginary part respectively):

  $$
  resultput_i = \alpha \cdot cos(\theta_i) + \beta \cdot sin(\theta_i) \\
  \alpha = \left\{
  \begin{aligned}
  1,\quad real(input_i) > 0 \\
  0,\quad real(input_i) = 0 \\
  -1,\quad real(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \beta = \left\{
  \begin{aligned}
  1,\quad image(input_i) > 0 \\
  0,\quad image(input_i) = 0 \\
  -1,\quad image(input_i) < 0 \\
  \end{aligned}
  \right.
  \\
  \theta_i = arctan(\frac{image(input_i)}{real(input_i)})
  $$

## Function Prototype

Each operator uses a [two-phase APIs](../../../docs/en/context/two_phase_api.md). Call "aclnnSignGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the computation process, and then call "aclnnSign" to execute the computation.

- `aclnnStatus aclnnSignGetWorkspaceSize(const aclTensor *self, aclTensor *result, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSign(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnSignGetWorkspaceSize

- **Parameters:**

  - self(const aclTensor, calculation input): tensor on the device, supporting 1 to 8 dimensions. Non-contiguous tensors are supported. The supported [data format](../../../docs/en/context/data_format.md) is ND.

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The supported data types include DOUBLE, FLOAT, FLOAT16, INT32, INT64, COMPLEX64, COMPLEX128, BOOL, and BFLOAT16.
    <!-- end id7 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: Supported data types include DOUBLE, FLOAT, FLOAT16, INT32, INT64, COMPLEX64, COMPLEX128, BOOL, and BFLOAT16.
    <!-- end id10 -->

  - result(aclTensor, calculation output): tensor on the device, supporting 1 to 8 dimensions. Non-contiguous tensors are supported. The supported [data format](../../../docs/en/context/data_format.md) is ND. The type and shape must be the same as those of self, and the dtype must be the same as that of self.

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The supported data types include DOUBLE, FLOAT, FLOAT16, INT32, INT64, COMPLEX64, COMPLEX128, BOOL, and BFLOAT16.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: Supported data types include DOUBLE, FLOAT, FLOAT16, INT32, INT64, COMPLEX64, COMPLEX128, BOOL, and BFLOAT16.
    <!-- end id9 -->

  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.
  - executor (aclOpExecutor**, output parameter): Returns the op executor, which includes computation.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 300px">
  <col style="width: 136px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self or result is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or data format of self or result is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and result do not match.</td>
    </tr>
    <tr>
      <td>The types of self and result do not match.</td>
    </tr>
  </tbody>
  </table>

## aclnnSign

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnSignGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
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

- Deterministic computation: aclnnSign defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sign.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (boilerplate) device/stream initialization. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7,8};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSign.
  ret = aclnnSignGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSignGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSign.
  ret = aclnnSign(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSign failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result in device-side memory to the host side. Change it according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Change it according to the definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources. Change it according to the definition of the specific API.
  aclrtFree(selfDeviceAddr);
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
