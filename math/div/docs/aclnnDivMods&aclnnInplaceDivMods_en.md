# aclnnDivMods&aclnnInplaceDivMods

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:53:43.966Z -->

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

- Description: Performs division calculation and selects the rounding operation based on the mode parameter.
- Formula:

$$
out_i = \frac{input_i}{other}
$$

## Function Prototype

aclnnDivMods and aclnnInplaceDivMods implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

- aclnnDivMods: Create a new output tensor object to store the computation result.
- aclnnInplaceDivMods: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDivModsGetWorkspaceSize" or "aclnnInplaceDivModsGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call "aclnnDivMods" or "aclnnInplaceDivMods" to perform the computation.

```Cpp
aclnnStatus aclnnDivModsGetWorkspaceSize(
  const aclTensor*   self,
  const aclScalar*   other,
  int                mode,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnDivMods(
  void*              workspace,
  uint64_t           workspaceSize,
  aclOpExecutor*     executor,
  aclrtStream        stream)
```

```Cpp
aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(
  aclTensor*         selfRef,
  const aclScalar*   other,
  int                mode,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnInplaceDivMods(
  void*              workspace,
  uint64_t           workspaceSize,
  aclOpExecutor*     executor,
  aclrtStream        stream)
```

## aclnnDivModsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 223px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 265px">
  <col style="width: 278px">
  <col style="width: 119px">
  <col style="width: 146px">
  <col style="width: 145px">
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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Dividend, corresponding to input in the formula.</td>
      <td>The data type must satisfy the data type derivation rule with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclScalar*)</td>
      <td>Input</td>
      <td>Divisor, corresponding to other in the formula.</td>
      <td>The data type and the data type of self must comply with the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode (int)</td>
      <td>Input</td>
      <td>Specifies the rounding mode of the quotient.</td>
      <td>The enumeration values are as follows:<br>0-corresponding to None: no rounding is performed by default;<br>1-corresponding to trunc: rounds the fractional part of the division result to zero;<br>2-corresponding to floor: rounds the division result down.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quotient, which is out in the formula.</td>
      <td>The data type must be a type that can be converted to the type derived from self and other. The shape is the same as that of self.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The data type must comply with the [TensorScalar deduction relationship](../../../docs/en/context/deduction_relationship.md).
    - When mode is 0, the data type of out supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, COMPLEX128, and COMPLEX64. If the data types of self and other after deduction are integer or Boolean types, the deducted data type is converted to FLOAT;
    - When mode is 1 or 2, the out data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16, COMPLEX128, and COMPLEX64.
  <!-- end id8 -->

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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of self or other is not within the supported range.</td>
    </tr>
    <tr>
      <td>self and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The dimensions of self and other are greater than 8.</td>
    </tr>
    <tr>
      <td>The value of mode is not within the range of 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>When mode is 1 or 2, the derived data type of self and other is COMPLEX64 or COMPLEX128.</td>
    </tr>
  </tbody>
  </table>

## aclnnDivMods

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnDivModsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceDivModsGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 223px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 265px">
  <col style="width: 278px">
  <col style="width: 119px">
  <col style="width: 146px">
  <col style="width: 145px">
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
      <td>selfRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Denotes the dividend and quotient, corresponding to input and out in the formula.</td>
      <td>The data type must be a data type that is convertible after the deduction between selfRef and other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclScalar*)</td>
      <td>Input</td>
      <td>Denotes the divisor, corresponding to other in the formula.</td>
      <td>The data type of this parameter and that of selfRef must comply with the data type derivation rule (for details, see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar Deduction Relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode (int)</td>
      <td>Input</td>
      <td>Specifies the rounding mode for the quotient.</td>
      <td>The enumerated values are as follows:<br>0-corresponding to None: no rounding is performed by default;<br>1-corresponding to trunc: rounds the decimal part of the division result toward zero;<br>2-corresponding to floor: rounds the division result down.</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id9 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The data type must satisfy the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">TensorScalar deduction relationship</a>.
    - When mode is 0 and the data type after derivation of selfRef and other is an integer type or Boolean type, the data type after derivation is converted to FLOAT.
    - When mode is 0, selfRef supports FLOAT, FLOAT16, DOUBLE, and BFLOAT16. When mode is 1 or 2, selfRef supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16. other does not support COMPLEX128 or COMPLEX64.
  <!-- end id10 -->

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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types and data formats of selfRef and other are not within the supported range.</td>
    </tr>
    <tr>
      <td>selfRef and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The number of dimensions of selfRef and other is greater than 8.</td>
    </tr>
    <tr>
      <td>The value of mode is not 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>When mode is 1 or 2, the data type derived from selfRef and other is COMPLEX64 or COMPLEX128.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceDivMods

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
      <td>Address of the workspace memory requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the Device, obtained by the first-phase API aclnnInplaceDivModsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDivMods&aclnnInplaceDivMods default to a deterministic implementation.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When mode is 1 (trunc mode) or 2 (floor mode), due to the limited precision of FLOAT16/BFLOAT16 data types, which cannot represent all decimals, certain errors may occur during rounding/flooring. You can choose a higher-precision data type such as FLOAT32.
<!-- end id11 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_div.h"

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
  // Call aclrtMalloc to apply for device-side memory.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction is custom based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outHostData(8, 0);
  float otherValue = 2.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclScalar.
  other = aclCreateScalar(&otherValue, aclDataType::ACL_FLOAT);
  CHECK_RET(other != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int mode = 2;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Example of calling the aclnnDivMods API
  LOG_PRINT("test aclnnDivMods\n");

  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the first-phase API of aclnnDivMods
  ret = aclnnDivModsGetWorkspaceSize(self, other, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDivMods
  ret = aclnnDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivMods failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Change it according to the specific API definition.
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]), outDeviceAddr,
                    8 * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < 8; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, outHostData[i]);
  }

  // Example of calling the aclnnInplaceDivMods API
  LOG_PRINT("\ntest aclnnInplaceDivMods\n");

  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the first-phase API of aclnnInplaceDivMods
  ret = aclnnInplaceDivModsGetWorkspaceSize(self, other, mode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivModsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnInplaceDivMods.
  ret = aclnnInplaceDivMods(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivMods failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyScalar(other);
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
