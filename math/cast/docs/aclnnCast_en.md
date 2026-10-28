# aclnnCast

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:37:52.502Z -->

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

Converts the input tensor to the specified dtype.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnCastGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnCast" API to execute the computation.

```Cpp
aclnnStatus aclnnCastGetWorkspaceSize(
  const aclTensor   *self,
  const aclDataType  dtype,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnCast(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCastGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1495px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 147px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the cast computation, which is an aclTensor on the Device side.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Target dtype to which the input tensor is converted.</td>
      <td>-</td>
      <td>const aclDataType</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter of the cast computation, an aclTensor on the Device side.</td>
      <td>Same shape as self.</td>
      <td>FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX32, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, INT4 (non-contiguous tensors not supported yet)</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: BFLOAT16 and INT4 are not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: COMPLEX32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, and INT4 are not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>tensor or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type or data format of self is not supported.</td>
    </tr>
    <tr>
      <td>The data format of self is different from that of out.</td>
    </tr>
    <tr>
      <td>The shape of self is different from the shape of out.</td>
    </tr>
    <tr>
      <td>The parameter dtype is not within the range of supported output data formats.</td>
    </tr>
    <tr>
      <td>When the data type of out is INT4, self is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>When the data type of out is INT4, the last dimension of the shape of self is odd.</td>
    </tr>
  </tbody></table>

## aclnnCast

- **Parameters:**
  <table style="table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 173px">
  <col style="width: 173px">
  <col style="width: 668px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace memory size allocated on the Device side, obtained by the first-phase API aclnnCastGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the operator computation process.</td>
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
  - aclnnCast defaults to a deterministic implementation.

- For scenarios where the data type is converted from floating-point to integer:
  If the input data contains NaN, NaN is converted to 0.

- For scenarios where the input data type is BOOL, COMPLEX32, COMPLEX64, COMPLEX128, FLOAT4_E2M1, or FLOAT4_E1M2:
  Non-contiguous input is not supported.

<!-- npu="A3,910b,910,310p" id9 -->
- <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - For the use case of converting the data type from int32 to int8: Only the input data in the range (-2048, 1920) is guaranteed to be error-free.
  - For the use case of converting the data type from float64/complex64/complex128 to uint8: Only the non-negative input data is guaranteed to be error-free.
<!-- end id9 -->
<!-- npu="310p" id10 -->
- <term>Atlas inference products</term>:
  - For the use case of converting data type from float32 to int64 and from float32 to uint8, precision is guaranteed only if the input data is within the range (-2147483648, 2147483583).
  - For the use case of converting data type from int64 to float32, precision is guaranteed only if the input data is within the range (-2147483648, 2147483647).
<!-- end id10 -->
<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>:
  - For scenarios where the output type is INT4: the last axis of the input shape being odd is not supported, and non-contiguous input is not supported.
  - For input and output types involving COMPLEX32, COMPLEX64, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, or INT4, only the conversion paths in the following table are supported:

    | `self` data type | `out` data type |
    | ------------ | ------------ |
    | COMPLEX32 | COMPLEX64 |
    | COMPLEX64 | COMPLEX32 |
    | COMPLEX32 | FLOAT16 |
    | FLOAT16 | COMPLEX32 |
    | FLOAT32/BFLOAT16 | COMPLEX64 |
    | COMPLEX64 | FLOAT32/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | FLOAT32/FLOAT16/BFLOAT16 |
    | FLOAT32/FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32/FLOAT16/BFLOAT16 |
    | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT4_E2M1/FLOAT4_E1M2 |
    | FLOAT4_E2M1/FLOAT4_E1M2 | HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
    | INT32 | INT4 |

<!-- end id11 -->

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"

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
  // Boilerplate, initialization.
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
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream by referring to the acl API documentation.
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

  std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  std::vector<double> outHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API aclnnCastGetWorkspaceSize.
  ret = aclnnCastGetWorkspaceSize(self, aclDataType::ACL_DOUBLE, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnCast.
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device memory to host memory, and modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor, and modify it based on the specific API definition.
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
