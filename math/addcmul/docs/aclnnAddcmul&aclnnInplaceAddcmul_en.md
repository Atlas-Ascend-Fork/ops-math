# aclnnAddcmul&aclnnInplaceAddcmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:14:04.115Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: supported
<!-- end id6 -->

## Function

- Description: Performs element-wise multiplication of tensor1 and tensor2, multiplies the result by the scalar value, and then performs element-wise addition with the input self/selfRef.
- Formula:

  $$
  out = self + value \times tensor1 \times tensor2
  $$

  Specifically, when using aclnnAddcmul, self in the formula corresponds to self in the first-phase API, and out in the formula corresponds to out in the first-phase API. When using aclnnInplaceAddcmul, self in the formula corresponds to selfRef in the first-phase API, and out in the formula corresponds to selfRef in the first-phase API.

## Function Prototype

- aclnnAddcmul and aclnnInplaceAddcmul implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnAddcmul: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceAddcmul: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnAddcmulGetWorkspaceSize" or "aclnnInplaceAddcmulGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the computation flow of the operator, and then call "aclnnAddcmul" or "aclnnInplaceAddcmul" to perform the computation.

  ```Cpp
  aclnnStatus aclnnAddcmulGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* tensor1,
    const aclTensor* tensor2,
    const aclScalar* value,
    aclTensor*       out,
    uint64_t*        workspaceSize,
  aclOpExecutor**    executor)
  ```

  ```Cpp
  aclnnStatus aclnnAddcmul(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(
    const aclTensor* selfRef,
    const aclTensor* tensor1,
    const aclTensor* tensor2,
    const aclScalar* value,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddcmul(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
  ```

## aclnnAddcmulGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 456px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
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
      <td>self</td>
      <td>Input</td>
      <td>The self in the formula.</td>
      <td>        <ul>
          <li>The data types of self, tensor1, and tensor2 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the derived type must be within the supported input types.</li>
          <li>The shapes of self, tensor1, and tensor2 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>Input</td>
      <td>The input tensor1 in the formula.</td>
      <td>        <ul>
          <li>The data types of tensor1, self, and tensor2 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the derived type must be one of the supported input types.</li>
          <li>The shapes of tensor1, self, and tensor2 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>Input</td>
      <td>Input tensor2 in the formula.</td>
      <td>        <ul>
          <li>The data types of tensor2, self, and tensor1 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the derived type must be one of the supported input types.</li>
          <li>The shapes of tensor2, self, and tensor1 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>The input value in the formula.</td>
      <td>The data type must be convertible to the derived data type of self, tensor1, and tensor2 (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output out in the formula.</td>
      <td>        <ul>
          <li>The data type must be convertible from the derived data types of self, tensor1, and tensor2 (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>The shape must be the same as the shape of self, tensor1, and tensor2 after broadcast.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, tensor1, tensor2, value, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or data format of self, tensor1, and tensor2 is not supported.</td>
    </tr>
    <tr>
      <td>self, tensor1, and tensor2 do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data type of self, tensor1, and tensor2 is not supported.</td>
    </tr>
    <tr>
      <td>The derived data type of self, tensor1, and tensor2 cannot be converted to the type of the specified output out.</td>
    </tr>
    <tr>
      <td>The shape of self, tensor1, or tensor2 exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shapes of self, tensor1, and tensor2 do not satisfy the broadcast relationship.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the shape of self, tensor1, and tensor2 after broadcast.</td>
    </tr>
  </tbody>
  </table>

## aclnnAddcmul

- **Parameters:**

  <table style="table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnAddcmulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddcmulGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1546px"><colgroup>
  <col style="width: 150px">
  <col style="width: 121px">
  <col style="width: 206px">
  <col style="width: 455px">
  <col style="width: 211px">
  <col style="width: 122px">
  <col style="width: 135px">
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
      <th>Dimension(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input/Output</td>
      <td>self/out in the formula.</td>
      <td>        <ul>
          <li>The data types of selfRef, tensor1, and tensor2 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), the derived data type can be converted to the data type of selfRef (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>), and the derived type must be within the supported input types.</li>
          <li>The shapes of selfRef, tensor1, and tensor2 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>, and the shape must be the same as the shape after broadcasting selfRef, tensor1, and tensor2.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor1</td>
      <td>Input</td>
      <td>Input tensor1 in the formula.</td>
      <td>        <ul>
          <li>The data types of tensor1, selfRef, and tensor2 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the derived type must be within the supported input types.</li>
          <li>The shapes of tensor1, selfRef, and tensor2 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tensor2</td>
      <td>Input</td>
      <td>Input tensor2 in the formula.</td>
      <td>        <ul>
          <li>The data types of tensor2, selfRef, and tensor1 satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>), and the derived type must be within the supported input types.</li>
          <li>The shapes of tensor2, selfRef, and tensor1 satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</li>
        </ul>
      </td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>The input value in the formula.</td>
      <td>The data type must be convertible to the data type derived from selfRef, tensor1, and tensor2 (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT8, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace that needs to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>selfRef, tensor1, tensor2, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types and data formats of selfRef, tensor1, and tensor2 are not supported.</td>
    </tr>
    <tr>
      <td>selfRef, tensor1, and tensor2 do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td>The derived data types of selfRef, tensor1, and tensor2 are not supported.</td>
    </tr>
    <tr>
      <td>The data type derived from selfRef, tensor1, and tensor2 cannot be converted to the type of the specified output selfRef.</td>
    </tr>
    <tr>
      <td>The shape of selfRef, tensor1, or tensor2 exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The shape of selfRef, tensor1, and tensor2 does not satisfy the broadcast relationship.</td>
    </tr>
    <tr>
      <td>The shape of selfRef is inconsistent with the shape of selfRef, tensor1, and tensor2 after broadcasting.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAddcmul

- **Parameters:**

  <table style="table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnInplaceAddcmulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation flow of the operator.</td>
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
  - aclnnAddcmul and aclnnInplaceAddcmul default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

aclnnAddcmul

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcmul.h"

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
  // Boilerplate: resource initialization.
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

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Fill in the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float scalarValue = 1.2f;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create value aclScalar.
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAddcmul.
  ret = aclnnAddcmulGetWorkspaceSize(self, tensor1, tensor2, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddcmul.
  ret = aclnnAddcmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddcmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyTensor(out);
  aclDestroyScalar(value);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
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

aclnnInplaceAddcmul

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addcmul.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization, see the acl API manual
  // Fill in deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API definition
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4, 2};
  std::vector<int64_t> tensor2Shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclScalar* value = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  float scalarValue = 1.2f;

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the tensor1 aclTensor.
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the tensor2 aclTensor.
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the value aclScalar.
  value = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceAddcmul.
  ret = aclnnInplaceAddcmulGetWorkspaceSize(self, tensor1, tensor2, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddcmul.
  ret = aclnnInplaceAddcmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddcmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyScalar(value);

  // 7.Release device resources. Modify them based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
