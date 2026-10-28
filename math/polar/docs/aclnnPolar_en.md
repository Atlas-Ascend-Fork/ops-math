# aclnnPolar

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:08:12.144Z -->

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

- Operator function: Converts the polar coordinates corresponding to the absolute value input and angle into Cartesian coordinates, producing a complex tensor out.

- Formula:

  $$
  out = abs*cos(angle) + abs*sin(angle)*j
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnPolarGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnPolar" API to perform the computation.

- `aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input, const aclTensor* angle, aclTensor* out,uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnPolar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnPolarGetWorkspaceSize

- **Parameters:**

  - input(aclTensor*, calculation input): Input tensor, which is `abs` in the formula. The data type supports FLOAT. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The number of dimensions cannot exceed 8.

  - angle(aclTensor*, calculation input): Input tensor, which is `angle` in the formula. Its data type must be the same as that of input. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The dimensions must be the same as those of input.

  - out(aclTensor*, calculation output): Output tensor, which is `out` in the formula. The data type supports COMPLEX64. [Non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) is supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The dimensions must be the same as those of input.
  - workspaceSize (uint64_t*, output parameter): Returns the workspace size to be allocated on the Device side.

  - executor (aclOpExecutor**, output parameter): Returns the op executor, which includes computation.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

```text
The first-phase API checks input parameters and reports an error in the following use cases:
Returns 161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input, angle, or out is a null pointer.
Returns 161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of input and angle are not within the supported range.
                                      2. The data types of input and angle are inconsistent.
                                      3. The data type of out is not COMPLEX64.
                                      4. The dimensions of input, angle, and out exceed 8 or are inconsistent.
```

## aclnnPolar

- **Parameters:**

  - workspace (void*, input parameter): Device-side allocated workspace memory address.

  - workspaceSize (uint64_t, input parameter): Device-side allocated workspace size, obtained from the first-phase API aclnnPolarGetWorkspaceSize.

  - executor (aclOpExecutor*, input parameter): Op executor, which includes the computation process of the operator.

  - stream (aclrtStream, input parameter): Stream on which the task is executed.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnPolar defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_polar.h"
#include <complex>

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
  // Boilerplate. Initialize resources.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // Set deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> inputShape = {1, 2};
  std::vector<int64_t> angleShape = {1, 2};
  std::vector<int64_t> outShape = {1, 2};
  void* inputDeviceAddr = nullptr;
  void* angleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* angle = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> inputHostData = {1, 2};
  std::vector<float> angleHostData = {1, 2};
  std::vector<std::complex<float>> outHostData = {1, 2};
  // Create the input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the angle aclTensor.
  ret = CreateAclTensor(angleHostData, angleShape, &angleDeviceAddr, aclDataType::ACL_FLOAT, &angle);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPolar.
  ret = aclnnPolarGetWorkspaceSize(input, angle, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPolarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPolar.
  ret = aclnnPolar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPolar failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5.Obtain the output values, copy the results from device memory to host memory, and modify them based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<std::complex<float>> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
   //The first number in the output is the real part, and the second number is the imaginary part.
   std::cout << "result[" << i << "]: " <<resultData[i] << std::endl;
  }

  // 6.Release aclTensor and aclIntArray. Modify the code based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(angle);
  aclDestroyTensor(out);

  // 7.Release device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(angleDeviceAddr);
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
