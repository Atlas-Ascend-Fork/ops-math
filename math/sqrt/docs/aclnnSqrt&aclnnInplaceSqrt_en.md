# aclnnSqrt&aclnnInplaceSqrt

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:17:30.186Z -->

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

- Description: Computes the non-negative square root. Returns NaN for negative inputs.

- Formula:

$$
out=sqrt(self)=\begin{cases}
\sqrt {self}, & self\ge 0 , \\
NaN, &  self\lt 0
\end{cases}
$$

## Function Prototype

- aclnnSqrt and aclnnInplaceSqrt implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnSqrt: Create a new output tensor object to store the computation result.
  - aclnnInplaceSqrt: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSqrtGetWorkspaceSize" or "aclnnInplaceSqrtGetWorkspaceSize" first to obtain input parameters and calculate the required workspace size based on the computation process, and then call "aclnnSqrt" or "aclnnInplaceSqrt" to perform the computation.

```cpp
aclnnStatus aclnnSqrtGetWorkspaceSize(
    const aclTensor* self,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  opExecutor)
```

```cpp
aclnnStatus aclnnSqrt(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* opExecutor,
    aclrtStream    stream)
```

```cpp
aclnnStatus aclnnInplaceSqrtGetWorkspaceSize(
    aclTensor*       self,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnInplaceSqrt(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnSqrtGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">data type</th>
      <th class="tg-0pky">data format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, self in the formula.</td>
      <td class="tg-0pky">shape must be the same as that of out.</td>
      <td class="tg-0pky">COMPLEX64, COMPLEX128, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor, out in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>If the input is complex, the output must be complex.</li>
          <li>The shape must be the same as that of self.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be requested on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">opExecutor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor containing the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1146px"><colgroup>
  <col style="width: 297px">
  <col style="width: 135px">
  <col style="width: 714px">
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
      <td>self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type, data format, or dimensions of self and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and out do not match.</td>
    </tr>
  </tbody>
  </table>

## aclnnSqrt

- **Parameters**

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
      <td>Size of the workspace memory allocated on the Device, obtained by the first-phase API aclnnSqrtGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>opExecutor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
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

## aclnnInplaceSqrtGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions(shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self(aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Input/output tensor, self/out in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor containing the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus, return status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1146px"><colgroup>
  <col style="width: 297px">
  <col style="width: 135px">
  <col style="width: 714px">
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
      <td>selfRef is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of selfRef is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceSqrt

- **Parameters**

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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnInplaceSqrtGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSqrt&aclnnInplaceSqrt default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sqrt.h"

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
  // Call aclrtMalloc to allocate memory on the device.
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
  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction as required by the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // aclnnSqrt API call example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSqrt.
  ret = aclnnSqrtGetWorkspaceSize(self, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSqrtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSqrt.
  ret = aclnnSqrt(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSqrt failed. ERROR: %d\n", ret); return ret);

  // aclnnInplaceSqrt API call example
  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // Call the first-phase API of aclnnInplaceSqrt.
  ret = aclnnInplaceSqrtGetWorkspaceSize(self, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSqrtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceSqrt.
  ret = aclnnInplaceSqrt(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSqrt failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from device memory to the host side. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnSqrt result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr, inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceSqrt result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. Release device resources, and modify according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (inplaceWorkspaceSize > 0) {
    aclrtFree(inplaceWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
