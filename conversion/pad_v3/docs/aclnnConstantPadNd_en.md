# aclnnConstantPadNd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:23:02.426Z -->

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

- Description: Pads the input tensor self with the constant value specified by value, based on the pad parameter.

- Formula:

  - Derivation formula for the shape of the out tensor:
    $
    \begin{aligned}
    &\text{Assume the shape of input self is:}
    \\ &[dim0_{in},dim1_{in}, dim2_{in}, dim3_{in}]
    &\text{Assume}
    \\ &pad =
    &[dim3_{begin},dim3_{end},
    \\&&dim2_{begin},dim2_{end},
    \\&&dim1_{begin},dim1_{end},
    \\&&dim0_{begin},dim0_{end}]
    \end{aligned}
    $

    $
    \begin{aligned}
    &\text{Then the shape of output out is:}
    \\ &[dim0_{out}, dim1_{out}, dim2_{out}, dim3_{out}] =
    &[dim0_{begin}+dim0_{in}+dim0_{end},
    \\&&dim1_{begin}+dim1_{in}+dim1_{end},
    \\&&dim2_{begin}+dim2_{in}+dim2_{end},
    \\&&dim3_{begin}+dim3_{in}+dim3_{end}]
    \end{aligned}
    $

  - Example 1:
    (When the pad array length equals twice the number of self dimensions)

    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\rbrace \\
    outputShape &= [8+1+9, 6+1+7, 4+1+5, 2+1+3, 0+1+1]\\
    &= [18,14,10,6,2]
    \end{aligned}
    $$

  - Example 2:
    (When the length of the pad array is less than twice the number of self dimensions)

    $$
    \begin{aligned}
    selfShape &= [1, 1, 1, 1, 1]\\
    pad &= \lbrace 0, 1, 2, 3, 4, 5\rbrace \\
    outputShape &= [0+1+0, 0+1+0, 4+1+5, 2+1+3, 0+1+1]\\
    &= [1,1,10,6,2]
    \end{aligned}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnConstantPadNdGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnConstantPadNd" to perform the computation.

```cpp
  aclnnStatus aclnnConstantPadNdGetWorkspaceSize(
    const aclTensor*   self,
    const aclIntArray* pad,
    const aclScalar*   value,
    aclTensor*         out,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
    aclnnStatus aclnnConstantPadNd(
      void*          workspace,
      uint64_t       workspaceSize,
      aclOpExecutor* executor,
      aclrtStream    stream)
```

## aclnnConstantPadNdGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>data type</th>
      <th>data format</th>
      <th>Dimension (shape)</th>
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Original input data to be padded.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pad (aclIntArray*)</td>
      <td>Input</td>
      <td>Dimensions to be padded on each axis of the input.</td>
      <td>The array length must be even and cannot exceed two dimensions of self.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value (aclScalar*)</td>
      <td>Input</td>
      <td>Padding value for the padded parts.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT6_E3M2, FLOAT6_E2M3, FLOAT4_E2M1, FLOAT4_E1M2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Padded output result.</td>
      <td>Same as self</td>
      <td>ND</td>
      <td>Same dimensions as self.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: data types BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: data types HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, and FLOAT4_E1M2 are not supported.
  <!-- end id8 -->
  - For CANN 9.0.0, when the data type of self is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT8_E8M0, only the case where all bits of value are 0 is supported.

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 724px">
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
        <td>self, pad, value, or out is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>Operator execution or variable initialization fails.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data types of self, value, or out are not within the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The data types of self and value do not meet the data type deduction rules.</td>
      </tr>
      <tr>
        <td>The shape derived from the shape of self and the input pad is inconsistent with the shape of out.</td>
      </tr>
      <tr>
        <td>An element in pad is not an even number, or the number of pad elements exceeds two dimensions of self.</td>
      </tr>
      <tr>
        <td>The dimension of self or out is greater than 8.</td>
      </tr>
      <tr>
        <td>Each value in pad must not make the shape of out less than 0. If pad contains positive numbers, the shape of out must not contain 0.</td>
      </tr>
      <tr>
        <td>When the data format of self is not ND, the data format of out is inconsistent with that of self.</td>
      </tr>
      <tr>
        <td>When the data type of self is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E8M0, FLOAT4_E2M1, or FLOAT4_E1M2, elements in pad are negative.</td>
      </tr>
      <tr>
        <td>For CANN 9.0.0, when the data type of self is HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT8_E8M0, value is not 0.</td>
      </tr>
    </tbody>
    </table>

## aclnnConstantPadNd

- **Parameters**
  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>The workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the Device, obtained by the first-phase API aclnnConstantPadNdGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
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

- Deterministic computation:
  - aclnnConstantPadNd defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <inttypes.h>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_constant_pad_nd.h"

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
  // Call aclrtMemcpy to copy data from the host to the device memory.
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

  // 2. Construct the input and output. Customize them based on the API.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> outShape = {5, 5};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* pad = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> outHostData(25, 0);
  float valueValue = 0.0f;
  std::vector<int64_t> padData = {1,1,1,1};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the pad array.
  pad = aclCreateIntArray(padData.data(), 4);
  CHECK_RET(pad != nullptr, return ret);
  // Create the value aclScalar.
  value = aclCreateScalar(&valueValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConstantPadNd.
  ret = aclnnConstantPadNdGetWorkspaceSize(self, pad, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnConstantPadNd.
  ret = aclnnConstantPadNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConstantPadNd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%" PRId64 "] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyIntArray(pad);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it according to the specific API definition.
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
