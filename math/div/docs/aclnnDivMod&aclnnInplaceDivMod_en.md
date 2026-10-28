# aclnnDivMod&aclnnInplaceDivMod

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:51:15.873Z -->

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
out_i = \frac{input_i}{other_i}
$$

## Function Prototype

aclnnDivMod and aclnnInplaceDivMod implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

- aclnnDivMod: Create a new output tensor object to store the computation result.
- aclnnInplaceDivMod: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDivModGetWorkspaceSize" or "aclnnInplaceDivModGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call "aclnnDivMod" or "aclnnInplaceDivMod" to perform computation.

```Cpp
aclnnStatus aclnnDivModGetWorkspaceSize(
  const aclTensor*   self,
  const aclTensor*   other,
  int                mode,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnDivMod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

```Cpp
aclnnStatus aclnnInplaceDivModGetWorkspaceSize(
  aclTensor*       selfRef,
  const aclTensor* other,
  int              mode,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceDivMod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnDivModGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>data type</th>
      <th>data format</th>
      <th>Dimension(shape)</th>
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self(aclTensor*)</td>
      <td>Input</td>
      <td>Dividend, corresponding to the input self in the formula.</td>
      <td><ul><li>The data type must satisfy the data type derivation rule with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>Up to 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Divisor, corresponding to the input other in the formula.</td>
      <td><ul><li>The data type must satisfy the data type derivation rule with the data type of self (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode (int)</td>
      <td>Input</td>
      <td>Specifies the rounding mode for the quotient.</td>
      <td>The enumeration values are as follows:<ul><li>0 - None: no rounding is performed by default.</li><li>1 - trunc: rounds the fractional part of the division toward zero.</li><li>2 - floor: rounds the division result down.</li></ul></td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quotient, that is, out in the formula.</td>
      <td><ul><li>The data type must be convertible from the data type derived from self and other.</li><li>The shape must be the shape after self and other are broadcast.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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
    - When mode is 0, the out data type supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, COMPLEX128, and COMPLEX64.
    - When mode is 1 or 2, the out data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16, COMPLEX128, and COMPLEX64.
    - When mode is 0 and the data type after derivation of self and other is an integer type or boolean type, the data type after derivation is converted to FLOAT.
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
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types and data formats of self and other are not within the supported range.</td>
    </tr>
    <tr>
      <td>self and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The dimensions of self and other are greater than 8.</td>
    </tr>
    <tr>
      <td>The value of mode is not in the range of 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>When mode is 1 or 2, the derived data types of self and other are COMPLEX64 or COMPLEX128.</td>
    </tr>
  </tbody>
  </table>

## aclnnDivMod

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
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnDivModGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. It contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceDivModGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1555px"><colgroup>
  <col style="width: 217px">
  <col style="width: 125px">
  <col style="width: 247px">
  <col style="width: 317px">
  <col style="width: 233px">
  <col style="width: 126px">
  <col style="width: 144px">
  <col style="width: 146px">
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
      <td>selfRef(aclTensor*)</td>
      <td>Input/Output</td>
      <td>Represents the dividend and quotient, corresponding to the inputs input and out in the formula.</td>
      <td><ul><li>The data type and the data type of other must satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>The data type must be a data type that can be converted after the derivation of selfRef and other (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li><li>The shape must satisfy the broadcast relationship with other (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>).</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the divisor, corresponding to the input other in the formula.</td>
      <td><ul><li>The data type must satisfy the data type derivation rule together with the data type of selfRef (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</li></ul></td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
      <td>Up to 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mode (int)</td>
      <td>Input</td>
      <td>Specifies the rounding mode for the quotient.</td>
      <td>The enumeration values are as follows:<ul><li>0-corresponds to None: no rounding is performed by default.</li><li>1-corresponds to trunc: rounds the fractional part of the division toward zero.</li><li>2-corresponds to floor: rounds the division result down.</li></ul></td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

  <!-- npu="910" id9 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When mode is 0, the selfRef data type supports FLOAT, FLOAT16, DOUBLE, and BFLOAT16.
    - When mode is 1 or 2, the selfRef data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, and BFLOAT16.
    - When mode is 0 and the data type after derivation of selfRef and other is an integer type or boolean type, the data type after derivation is converted to FLOAT.
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
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types and data formats of selfRef and other are not within the supported range.</td>
    </tr>
    <tr>
      <td>selfRef and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The dimensions of selfRef and other are greater than 8.</td>
    </tr>
    <tr>
      <td>The value of mode is not in the range of 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>When mode is 1 or 2, the data type derived from selfRef and other is COMPLEX64 or COMPLEX128.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceDivMod

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by calling the first-phase API aclnnInplaceDivModGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDivMod&aclnnInplaceDivMod defaults to a deterministic implementation.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When mode is 1 (trunc mode) or 2 (floor mode), due to the limited precision of FLOAT16/BFLOAT16 data types, which cannot represent all decimals, certain errors may occur during rounding/flooring. You can choose a higher-precision data type such as FLOAT32.
<!-- end id11 -->

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int mode = 2;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnDivMod API call example
  LOG_PRINT("test aclnnDivMod\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnDivMod
  ret = aclnnDivModGetWorkspaceSize(self, other, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivModGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDivMod
  ret = aclnnDivMod(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDivMod failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceDivMod API call example
  LOG_PRINT("\ntest aclnnInplaceDivMod\n");

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  // Call the first-phase API of aclnnInplaceDivMod
  ret = aclnnInplaceDivModGetWorkspaceSize(self, other, mode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivModGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnInplaceDivMod second-phase API.
  ret = aclnnInplaceDivMod(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivMod failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from the device-side memory to the host side. Modify it based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);

  // 7. Release the device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
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
