# aclnnComplex

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:42:01.501Z -->

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

- Description: Takes as input two tensors of the same Dtype whose shapes satisfy the broadcast relationship.
- Formula:

  $$
  \text{out}[i] = \text{real}[i] + \text{imag}[i]*\mathrm{j}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnComplexGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnComplex" API to perform the computation.

```cpp
aclnnStatus aclnnComplexGetWorkspaceSize(
  const aclTensor*    real,
  const aclTensor*    imag,
  aclTensor*          out,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnComplex(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

## aclnnComplexGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 140px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>real (aclTensor*)</td>
        <td>Input</td>
        <td>The input real in the formula represents the real part of a complex number.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with imag.</td>
        <td>FLOAT, FLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>imag (aclTensor*)</td>
        <td>Input</td>
        <td>The input imag in the formula represents the imaginary part of a complex number.</td>
        <td>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with real.</td>
        <td>FLOAT, FLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>out in the formula, a tensor of complex types.</td>
        <td>shape must be the shape after broadcasting real and imag.</td>
        <td>COMPLEX64, COMPLEX32, COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="950,A3" id7 -->
  - <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: The out data type supports COMPLEX64 (the input must be FLOAT), COMPLEX32 (the input must be FLOAT16), and COMPLEX128 (the input must be DOUBLE).
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
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
      <td>The real, imag, or out pointer is null.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type and data format of real and imag are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of real and imag cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The dimensions of real and imag are greater than 8.</td>
    </tr>
    <tr>
      <td>The data types of real and imag are different.</td>
    </tr>
  </tbody>
  </table>

## aclnnComplex

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
        <td>Address of the workspace memory allocated on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace memory allocated on the Device, obtained by calling the first-phase API aclnnComplexGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnComplex defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <complex>
#include "acl/acl.h"
#include "aclnnop/aclnn_complex.h"

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
  // Boilerplate: resource initialization
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> realShape = {4, 2};
  std::vector<int64_t> imagShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* realDeviceAddr = nullptr;
  void* imagDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* real = nullptr;
  aclTensor* imag = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> realHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> imagHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<std::complex<float>> outHostData(8, 0);
  // Create the real aclTensor.
  ret = CreateAclTensor(realHostData, realShape, &realDeviceAddr, aclDataType::ACL_FLOAT, &real);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the imag aclTensor.
  ret = CreateAclTensor(imagHostData, imagShape, &imagDeviceAddr, aclDataType::ACL_FLOAT, &imag);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_COMPLEX64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnComplex.
  ret = aclnnComplexGetWorkspaceSize(real, imag, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnComplex.
  ret = aclnnComplex(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplex failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait until the task execution ends.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(real);
  aclDestroyTensor(imag);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(realDeviceAddr);
  aclrtFree(imagDeviceAddr);
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
