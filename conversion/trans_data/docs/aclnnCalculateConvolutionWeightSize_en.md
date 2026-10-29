# aclnnCalculateConvolutionWeightSize

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:39:05.367Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Calculates the memory size required for the weight tensor of the Convolution operator under the NCHW format input. Only the Float16 data type is supported. This API is used solely to determine the size needed for preprocessing the weight tensor to achieve optimal performance of the Convolution operator.

- Example: For an input of [2, 4, 8, 8], the function changes the shape to [64, 1, 16, 16] for performance reasons, and therefore modifies the input to 16384.

## Function Prototype

```cpp
aclnnStatus aclnnCalculateConvolutionWeightSize(
    const aclIntArray* tensorShape,
    bool               transposed,
    int64_t            groups,
    aclDataType        dataType,
    uint64_t*          weightTensorSize)
```

## aclnnCalculateConvolutionWeightSize

- **Parameters:**

  <table>
  <tr>
  <th style="width:211px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:266px">Description</th>
  <th style="width:308px">Instruction</th>
  <th style="width:240px">Data Type</th>
  <th style="width:110px">Data Format</th>
  <th style="width:150px">Dimension (shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>tensorShape (aclIntArray*)</td>
  <td>Input</td>
  <td>Indicates the shape of the weight matrix loaded for this Convolution.</td>
  <td>Only 4-dimensional shapes in NCHW format are supported, and each dimension must be &gt;= 0. Empty tensors are supported, and weightTensorSize is returned as 0.</td>
  <td>INT64</td>
  <td>NCHW</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>transposed (bool)</td>
  <td>Input</td>
  <td>Indicates whether it is transposed convolution.</td>
  <td>Currently, only false is supported.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>groups (int64_t)</td>
  <td>Input</td>
  <td>Represents the number of blocked connections from input channels to output channels.</td>
  <td>Value range: [1,65535].</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>dataType (aclDataType)</td>
  <td>Input</td>
  <td>Indicates the data type of the converted weight.</td>
  <td>Only ACL_FLOAT16 is supported.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>weightTensorSize (uint64_t*)</td>
  <td>Output</td>
  <td>Calculates the amount of data (in elements) required by weight for this input based on the internal processing logic of Convolution.</td>
  <td>-</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Values</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">Input shape verification fails or other inputs do not meet expectations.</td>
    </tr>
  </tbody>
  </table>

## Constraints

- deterministic computation:
  - aclnnCalculateConvolutionWeightSize defaults to a deterministic implementation.

- Only the forward Conv2D scenario is supported.
- Transposed convolution is not supported.
- Empty tensors are supported: weightTensorSize is returned as 0.

## Example

The sample code is as follows and is for reference only. For the specific compilation and execution process, see <a href="../../../docs/en/context/compile_and_run_sample.md">Compile and Run Samples</a>.

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_trans_convolution_weight.h"
using namespace std;
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
  // Boilerplate. Initialize the stream.
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
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateWeightAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, uint64_t &TransWeightSize)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call the transweight host API to calculate the actual number of elements.
    const aclIntArray* weightSize = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateConvolutionWeightSize(weightSize, false, 1, aclDataType::ACL_FLOAT16, &TransWeightSize);
    // Call aclrtMalloc to allocate device memory.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    // Call aclrtMemcpy to copy Host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              return ret);
    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);

    return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> inputShape = {1, 4, 16, 16};
  std::vector<int64_t> weightShape = {2, 4, 8, 8};
  std::vector<int64_t> biasShape = {2};
  std::vector<int64_t> outShape = {1, 2, 9, 9};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  // aclTensor* transWeight = nullptr;
  std::vector<float> inputHostData(1024, 1);
  std::vector<float> weightHostData(512, 1);
  std::vector<float> biasHostData(2, 1);
  std::vector<float> outHostData(162, 0);
  uint64_t transWeightSize = 0;

  // Create the self aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateWeightAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16,
    &weight, transWeightSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the Transweight acltensor.
  void* transWeightDeviceAddr = nullptr;
  uint64_t size = transWeightSize * sizeof(float) / 2;
  ret = aclrtMalloc(&transWeightDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);return ret);

  std::vector<float> transData;
  transData.resize(transWeightSize * 2);

  // Call aclrtMemcpy to copy host-side data to the device memory transData.data().
  ret = aclrtMemcpy(transWeightDeviceAddr, size, transData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of the contiguous tensor.
  vector<int64_t> shape = weightShape;
  std::vector<int64_t> s(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      s[i] = shape[i + 1] * s[i + 1];
  }

  aclTensor* transWeight = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, s.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                shape.data(), shape.size(), transWeightDeviceAddr);

  // 3. Call the CANN operator library API, and change it to the specific API name.
  int8_t cubeMathType = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  bool transposed = 0;
  uint64_t groups = 1;
  // Call TransWeight.
  ret = aclnnTransConvolutionWeightGetWorkspaceSize(weight, transposed, groups, transWeight,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnTransConvolutionWeight.
  ret = aclnnTransConvolutionWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransConvolutionWeight failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> convStrides = {1, 1, 1, 1};
  std::vector<int64_t> convPads = {0, 0, 0, 0};
  std::vector<int64_t> convOutPads = {1, 1, 1, 1};
  std::vector<int64_t> convDilations = {1, 1, 1, 1};

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);

  // 3. Call the CANN operator library API, and change it to the specific API.
  workspaceSize = 0;
  // Call the first-phase API aclnnConvolution.
  ret = aclnnConvolutionGetWorkspaceSize(input, transWeight, bias, strides, pads, dilations, false, outPads, groups,
    out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnConvolution.
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device-side memory to the host side. Modify it based on the definition of the specific API.
  size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the definition of the specific API.
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(transWeight);
  aclDestroyTensor(bias);
  aclDestroyTensor(out);

  aclDestroyIntArray(strides);
  aclDestroyIntArray(pads);
  aclDestroyIntArray(outPads);
  aclDestroyIntArray(dilations);

  // 7. Release device resources. Modify it based on the definition of the specific API.
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(transWeightDeviceAddr);
  aclrtFree(biasDeviceAddr);
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
