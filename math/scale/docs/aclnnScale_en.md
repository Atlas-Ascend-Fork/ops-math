# aclnnScale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:59:54.361Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Computes the result of scale and bias on the input.

- Formula:

  If bias is not provided, then

  $$
  y=x*scale
  $$

  If bias is input, then

  $$
  y=x*scale + bias
  $$

  NOTE
scale/bias supports broadcasting with x. The shape rules for scale/bias are as follows:

  - When scaleFromBlob is True (axis is converted to a positive number, and numAxes being -1 indicates up to the last axis):

    When numAxes=0, scale must contain exactly one element, with rank 0 through 8; (), [1], and [1,1] are scalar forms. x must still have rank 1 through 8 and a valid axis.

    When numAxes=-1, scaleShape is xShape[axis:]; when numAxes>0, axis + numAxes must not exceed rank(xShape), and scaleShape is xShape[axis:axis + numAxes]. If bias is provided, its shape and dtype must equal those of scale.

  - When scaleFromBlob is False (axis is normalized; numAxes does not determine the shape but must still be >=-1):

    scaleShape is xShape[axis:axis + rank(scaleShape)]

    biasShape is xShape[axis:axis + rank(scaleShape)]

    axis + rank(scaleShape) must not exceed rank(xShape). () and [1] use scalar broadcasting; other scale shapes must match the selected x window dimension by dimension. A size-1 dimension cannot replace a non-1 dimension in that window. Bias, when present, must have the same shape as scale.
  - Example:

    - scaleFromBlob = True:

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = 2  --> scaleShape = [d, e]

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = 3  --> scaleShape = [d, e, f]

      xShape = [a, b, c, d, e, f] axis = 3 numAxes = -1 --> scaleShape = [d, e, f]

    - scaleFromBlob = False:

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 2 --> scaleShape = [d, e]

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 3 --> scaleShape = [d, e, f]

      xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 1 --> scaleShape = [d]

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnScaleGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call the "aclnnScale" API to perform computation.

```Cpp
aclnnStatus aclnnScaleGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scale,
  const aclTensor* bias,
  int64_t          axis,
  int64_t          numAxes,
  bool             scaleFromBlob,
  aclTensor*       y,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScale(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScaleGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 190px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 160px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td>Tensor for the operator input.</td>
      <td>Must not be a null pointer. Shape dimensions must be nonnegative. Empty tensors have at least one zero-length dimension; all shape, dtype, and axis constraints are checked before returning an empty output with the same shape. A rank-0 x has no valid axis and is rejected.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (const aclTensor*)</td>
      <td>Input</td>
      <td>Tensor for the operator input.</td>
      <td>Must not be a null pointer. The dtype must equal x. Shape dimensions must be nonnegative and satisfy the strict window rules (see <a href="#function">Function</a>). Zero-element scale is valid only when it matches a zero-length x window.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Tensor for operator input.</td>
      <td>nullptr omits bias. A present tensor, including a zero-element tensor, must have the same dtype and shape as scale.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Specifies the starting axis for scale.</td>
      <td>Value range [-x_rank, x_rank) (x_rank indicates the number of dimensions of x).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numAxes (int64_t)</td>
      <td>Input</td>
      <td>Specifies the number of axes on which scale is performed.</td>
      <td>Both branches require numAxes>=-1. Only when scaleFromBlob=True does -1 select the axes from axis to the last axis; False derives its window from scale rank.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleFromBlob (bool)</td>
      <td>Input</td>
      <td>Selects how the scale/bias broadcast window is derived.</td>
      <td><ul><li>True: scale from blob, use numAxes + axis to perform scaling;</li><li>False: scale from input scale, scale the input scale length starting from axis, and do not use numAxes to derive the window (numAxes must still be >=-1).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Must not be a null pointer. Supports empty tensors. The data type must match x, and the shape must exactly match x in both rank and the length of every dimension.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>Must be a valid non-null output address. Use the workspace size returned by this API; it may include temporary storage for contiguous input copies, intermediate outputs, and output copy-back.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>Must be a valid non-null output address.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: BFLOAT16 is not supported.
  <!-- end id7 -->

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
      <th>Return Code</th>
      <th>Value</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, scale, y, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type of x is not within the supported range.</td>
    </tr>
    <tr>
      <td>If bias is provided, the data types of bias and scale are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of scale and x are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of y and x are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of x and y are inconsistent.</td>
    </tr>
    <tr>
      <td>If bias is provided, the shapes of bias and scale are inconsistent.</td>
    </tr>
    <tr>
      <td>x has rank outside 1 through 8, scale or bias has rank greater than 8, or a shape of x, scale, bias, or y contains a negative dimension.</td>
    </tr>
    <tr>
      <td>The value of axis is not within the range [-x_rank, x_rank).</td>
    </tr>
    <tr>
      <td>The value of numAxes is less than -1.</td>
    </tr>
    <tr>
      <td>scaleFromBlob is True, numAxes is 0, and scale does not contain exactly one element.</td>
    </tr>
    <tr>
      <td>When scaleFromBlob=True and numAxes>0, the normalized axis plus numAxes exceeds x_rank.</td>
    </tr>
    <tr>
      <td>The shape of scale does not match the expected shape (see Function for the derivation of the expected shape).</td>
    </tr>
  </tbody>
  </table>

## aclnnScale

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by the first-phase API aclnnScaleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. It contains the operator computation flow.</td>
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

- aclnnScale defaults to a deterministic implementation.
- x, scale, bias, and y must have the same data type. A nullptr bias is excluded from this check.
- The shape of y must exactly match x in both rank and the length of every dimension.
- When bias is not nullptr, its shape must exactly match scale.
- The shape of scale must satisfy the broadcast window rules in [Function](#function).
- [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are supported, with at most eight dimensions.
- ND in the parameter table is the publicly supported format. The interface does not uniformly reject every non-ND descriptor; the framework may convert such a descriptor before execution. This does not guarantee support for arbitrary non-ND format combinations.
- Executable tensor sizes depend on available device memory and framework resources. This interface does not specify a universal limit of 2³¹ elements or 4 GiB.

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: FLOAT16 and BFLOAT16 inputs compute in FLOAT, then convert back to the input data type. FLOAT inputs compute directly.
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scale.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet) {
  // Boilerplate: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  initialized = true;
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  deviceSet = true;
  aclrtStream rawStream = nullptr;
  ret = aclrtCreateStream(&rawStream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  stream.reset(rawStream);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device memory.
  void* rawDeviceAddr = nullptr;
  auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  deviceAddr.reset(rawDeviceAddr);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                       aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
  CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
  tensor.reset(rawTensor);
  return 0;
}

int main() {
  // 1. (boilerplate) device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  bool initialized = false;
  bool deviceSet = false;
  std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
    if (deviceSet) {
      aclrtResetDevice(deviceId);
    }
    if (initialized) {
      aclFinalize();
    }
  });
  StreamPtr stream(nullptr, &aclrtDestroyStream);
  auto ret = Init(deviceId, stream, initialized, deviceSet);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4};
  std::vector<int64_t> tensor2Shape = {4};
  std::vector<int64_t> outShape = {4, 2};
  DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr tensor1DeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr tensor2DeviceAddr(nullptr, &aclrtFree);
  DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
  TensorPtr self(nullptr, &aclDestroyTensor);
  TensorPtr tensor1(nullptr, &aclDestroyTensor);
  TensorPtr tensor2(nullptr, &aclDestroyTensor);
  TensorPtr out(nullptr, &aclDestroyTensor);

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t axis = 0;
  int64_t numAxes = 1;
  bool fromBlob = true;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, aclDataType::ACL_FLOAT, tensor1DeviceAddr, tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, aclDataType::ACL_FLOAT, tensor2DeviceAddr, tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScale.
  ret = aclnnScaleGetWorkspaceSize(self.get(), tensor1.get(), tensor2.get(), axis, numAxes, fromBlob, out.get(),
                                   &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize computed by the first-phase API.
  DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
  if (workspaceSize > 0) {
    void* rawWorkspaceAddr = nullptr;
    ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddr.reset(rawWorkspaceAddr);
  }
  // Call the second-phase API of aclnnScale.
  ret = aclnnScale(workspaceAddr.get(), workspaceSize, executor, stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScale failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream.get());
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr.get(), size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  return 0;
}
```
