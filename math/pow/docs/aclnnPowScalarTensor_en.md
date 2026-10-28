# aclnnPowScalarTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:15:13.231Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Each element of exponent is used as the exponent for the corresponding element of input to perform the power calculation.

- Formula:

$$
out_i = x_i^{exponent_i}
$$

- Operator constraint: INT32 integer computation times out in scenarios outside the following ranges.

  |shape|exponent_value|
  |----|----|
  |<=100,000 (one hundred thousand)|-200,000,000~200,000,000 (two hundred million)|
  |<=1,000,000 (one million)|-20,000,000~20,000,000 (twenty million)|
  |<=10,000,000 (ten million)|-2,000,000~2,000,000 (two million)|
  |<=100,000,000 (one hundred million)|-200,000~200,000 (two hundred thousand)|
  |<=1,000,000,000 (one billion)|-20,000~20,000 (twenty thousand)|

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnPowScalarTensorGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the computation process, and then call "aclnnPowScalarTensor" to execute the computation.

```Cpp
aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(
 const aclScalar* self,
 const aclTensor* exponent,
 const aclTensor* out,
 uint64_t*        workspaceSize,
 aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnPowScalarTensor(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnPowScalarTensorGetWorkspaceSize

- **Parameters**

    <table style="table-layout: fixed; width: 1400px"><colgroup>
    <col style="width: 155px">
    <col style="width: 120px">
    <col style="width: 248px">
    <col style="width: 223px">
    <col style="width: 229px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data type</th>
        <th>Data format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self (aclScalar*)</td>
        <td>Input</td>
        <td>Base of the pow operation, x in the formula.</td>
        <td>The data type and exponent satisfy the <a href ="../../../docs/en/context/deduction_relationship.md">TensorScalar deduction relationship</a>.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>exponent (aclTensor*)</td>
        <td>Input</td>
        <td>Exponent of the pow operation, exponent in the formula.</td>
        <td>The data type and self satisfy the <a href ="../../../docs/en/context/deduction_relationship.md">TensorScalar deduction relationship</a>.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>No more than 8 dimensions</td>
        <td>√</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output of the pow operation, which is out in the formula.</td>
        <td>The data type must be convertible from the data types of self and exponent after deduction (see <a href ="../../../docs/en/context/conversion_relationship.md">conversion relationship</a>).</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT16, INT32, INT64, INT8, UINT8, COMPLEX64, COMPLEX128, BFLOAT16, UINT16, UINT32, UINT64</td>
        <td>ND</td>
        <td>Same as exponent.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
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

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type BFLOAT16 is not supported.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The self, exponent, and out are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of exponent is not within the supported range.</td>
      </tr>
      <tr>
      <td>self and exponent do not comply with the data type deduction rules.</td>
      </tr>
      <tr>
      <td>The deduced data type cannot be converted to the type of out.</td>
      </tr>
      <tr>
      <td>The shapes of exponent and out are inconsistent.</td>
      </tr>
    </tbody>
    </table>

## aclnnPowScalarTensor

- **Parameters**

    <table style="table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Address of the workspace memory requested on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the device, obtained by calling the first-phase API aclnnPowScalarTensorGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnPowScalarTensor defaults to a deterministic implementation.
- The data types of self and exponent cannot be both BOOL.

<!-- npu="910,310p" id8 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: In this scenario, if the calculated result exceeds the value range of the specified data type, the boundary value of that data type is returned as the result.
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"

#define CHECK_RET(cond, return_exponentr) \
  do {                               \
    if (!(cond)) {                   \
      return_exponentr;                   \
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
  // Boilerplate for resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
  // 1. (boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> exponentShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* exponentDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclScalar* self = nullptr;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> exponentHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float selfValue = 1.2f;
  // Create the self aclScalar.
  self = aclCreateScalar(&selfValue, aclDataType::ACL_FLOAT);
  CHECK_RET(self != nullptr, return ret);
  // Create the exponent aclTensor.
  ret = CreateAclTensor(exponentHostData, exponentShape, &exponentDeviceAddr, aclDataType::ACL_FLOAT, &exponent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace the API name with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPowScalarTensor.
  ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPowScalarTensor.
  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowScalarTensor failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from device memory to host memory. Modify as required by the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify as required by the API definition.
  aclDestroyScalar(self);
  aclDestroyTensor(exponent);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify as required by the API definition.
  aclrtFree(exponentDeviceAddr);
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
