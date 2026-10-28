# aclnnDigamma

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:47:32.854Z -->

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

- Operator function: Performs the digamma mathematical operation on each element of the input tensor.

- Formula:

$$
out_i= \frac{{\rm d}} {{\rm d}self_i}log_e(\Gamma(|self_i|))
$$

## Function Prototype

Each operator uses a [two-phase API](./../../../docs/en/context/two_phase_api.md). Call "aclnnDigammaGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call "aclnnDigamma" to execute the computation.

```Cpp
aclnnStatus aclnnDigammaGetWorkspaceSize(
  const aclTensor *self,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDigamma(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDigammaGetWorkspaceSize

- **Parameters:**

  | Name | Input/Output | Description |
  | --- | --- | --- |
  | self | Input | aclTensor on the device, with supported data types FLOAT, FLOAT16, and DOUBLE. Supports [non-contiguous tensors](./../../../docs/en/context/non_contiguous_tensor.md). Supported [data format](./../../../docs/en/context/data_format.md) is ND. |
  | out | Output | aclTensor on the device, with supported data types FLOAT, FLOAT16, and DOUBLE. Supports [non-contiguous tensors](./../../../docs/en/context/non_contiguous_tensor.md). Supported [data format](./../../../docs/en/context/data_format.md) is ND. |
  | workspaceSize | Output | Returns the workspace size to be applied on the Device side. |
  | executor | Output | Returns the op executor, which contains the operator computation flow. |

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](./../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API validates input parameters and reports errors in the following use cases:
Returns 161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input tensor or out is a null pointer.
Returns 161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types and data formats of self and out are not within the supported range.
                                      2. The shapes, data formats, or data types of self and out are inconsistent.
                                      3. The number of dimensions of self and out is greater than 8
```

## aclnnDigamma

- **Parameters:**

  | Name | Input/Output | Description |
  | --- | --- | --- |
  | workspace | Input | Device-side allocated workspace memory address. |
  | workspaceSize | Input | Size of the workspace requested on the Device, obtained by the first-phase API aclnnDigammaGetWorkspaceSize. |
  | executor | Input | Op executor that contains the operator computation flow. |
  | stream | Input | Stream on which the task is executed. |

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn return codes](./../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDigamma defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](./../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_digamma.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate code for resource initialization
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
  // Call aclrtMalloc to apply for memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1, -1, -1, -2, 2, -2, -3, 3};
  std::vector<float> outHostData = {1, -1, -1, -2, 2, -2, -3, 3};

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnDigamma.
  ret = aclnnDigammaGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDigammaGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnDigamma.
  ret = aclnnDigamma(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDigamma failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from the device to the host. Modify it based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources.
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
