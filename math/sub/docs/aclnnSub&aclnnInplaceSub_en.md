# aclnnSub&aclnnInplaceSub

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:23:52.404Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs subtraction, with the minuend scaled by alpha.
- Formula:

  $$
  out_{i} = self_{i} - alpha \times other_{i}
  $$

  $$
  selfRef_{i}=selfRef_{i}-alpha \times other_{i}
  $$

## Function Prototype

- aclnnSub and aclnnInplaceSub implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnSub: Create a new output tensor object to store the computation result.
  - aclnnInplaceSub: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSubGetWorkspaceSize" or "aclnnInplaceSubGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnSub" or "aclnnInplaceSub" to perform the computation.

```cpp
aclnnStatus aclnnSubGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* other,
    const aclScalar* alpha,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnSub(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

```cpp
aclnnStatus aclnnInplaceSubGetWorkspaceSize(
    aclTensor* selfRef,
    const aclTensor* other,
    const aclScalar* alpha,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnInplaceSub(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnSubGetWorkspaceSize

- **Parameters**:

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
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">self in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must satisfy the data type derivation rule with the data type of other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">other in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type and the data type of self must satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">alpha (aclScalar*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Alpha in the formula.</td>
      <td class="tg-0pky">The data type must be convertible to the derived data type of self and other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">out in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type must be a data type that self and other can be converted to after deduction (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>The shape must be the shape after self and other are <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the Device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310b" id7 -->
  - <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).
  The first-phase API validates input parameters. Errors are reported in the following cases:

  <table style="table-layout: fixed; width: 1142px"><colgroup>
  <col style="width: 305px">
  <col style="width: 138px">
  <col style="width: 699px">
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
      <td>When self, other, alpha, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of self and other are not in the supported range.</td>
    </tr>
    <tr>
      <td>self and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>alpha cannot be converted to the derived data type of self and other.</td>
    </tr>
    <tr>
      <td>The shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape of out is not the broadcast shape of self and other.</td>
    </tr>
    <tr>
      <td>The shape of self or other has more than 8 dimensions.</td>
    </tr>
  </tbody>
  </table>

## aclnnSub

- **Parameters**:

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnSubGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceSubGetWorkspaceSize

- **Parameters**:

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
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">selfRef (aclTensor*)</td>
      <td class="tg-0pky">Input|Output</td>
      <td class="tg-0pky">selfRef in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type of selfRef must satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>) and must be a data type that can be converted after derivation (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>The shape of selfRef must be the same as the shape of other after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">other (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">other in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>The data type needs to satisfy the data type derivation rule with that of selfRef (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li>
          <li>The shape needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">alpha (aclScalar*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">alpha in the formula.</td>
      <td class="tg-0pky">The data type needs to be convertible to the derived data types of selfRef and other (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128, COMPLEX64, BFLOAT16</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
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
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310b" id8 -->
  - <term>Atlas training products</term> and <term>Atlas 200I/500 A2 inference products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 298px">
  <col style="width: 135px">
  <col style="width: 716px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Codes</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>When selfRef, other, or alpha is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of selfRef and other are not in the supported range.</td>
    </tr>
    <tr>
      <td>selfRef and other do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of selfRef.</td>
    </tr>
    <tr>
      <td>alpha cannot be converted to the derived data types of selfRef and other.</td>
    </tr>
    <tr>
      <td>The shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>The shape after broadcast is not equal to the shape of selfRef.</td>
    </tr>
    <tr>
      <td>The dimensions of selfRef or other are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceSub

- **Parameters**:

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device. The value is obtained by the first-phase API aclnnInplaceSubGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSub&aclnnInplaceSub default to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sub.h"

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
  // Call aclrtMalloc to allocate memory on the Device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host data to Device memory.
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
  // 1. (Boilerplate) Initialize device/stream. See the list of external acl APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Construct them as required by the API.
  std::vector<int64_t> shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float alphaValue = 1.2f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the other aclScalar.
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Example of calling the aclnnSub API.
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnSub first-phase API.
  ret = aclnnSubGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSubGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnSub second-phase API.
  ret = aclnnSub(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSub failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // Example of calling the aclnnInplaceSub API.
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceSub\n");
  // Call the first-phase API of aclnnInplaceSub.
  ret = aclnnInplaceSubGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSubGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceSub.
  ret = aclnnInplaceSub(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSub failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result on the device memory to the host side. Modify the code based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);
  // 7. Release device resources. Modify the code based on the specific API definition.
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
