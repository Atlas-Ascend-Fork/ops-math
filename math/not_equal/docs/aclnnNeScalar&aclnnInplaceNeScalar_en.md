# aclnnNeScalar&aclnnInplaceNeScalar

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:07:33.734Z -->

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

- Description: Computes whether the value of each element in selfRef is not equal to the value of other.
- Formula:

$$
out_i​=(self_i \ne other)?[1]: [0]
$$

$$
selfRef_i​=(selfRef_i \ne other)?[1]: [0]
$$

## Function Prototype

- aclnnNeScalar and aclnnInplaceNeScalar implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnNeScalar: Create a new output tensor object to store the computation result.
  - aclnnInplaceNeScalar: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNeScalarGetWorkspaceSize" or "aclnnInplaceNeScalarGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow. Then, call "aclnnNeScalar" or "aclnnInplaceNeScalar" to perform the computation.

```Cpp
aclnnStatus aclnnNeScalarGetWorkspaceSize(
    const aclTensor *self,
    const aclScalar *other,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnNeScalar(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceNeScalarGetWorkspaceSize(
    aclTensor       *selfRef,
    const aclScalar *other,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceNeScalar(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnNeScalarGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>Non-contiguous is supported.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclScalar*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>The out in the formula.</td>
      <td>The data type of out must be a data type that BOOL can be converted to.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64, UINT32, UINT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - self and other do not support UINT64.
    - out does not support UINT64, UINT32, or UINT16.
  <!-- end id7 -->
  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>:
    - self and other do not support BFLOAT16 or UINT64.
    - out does not support BFLOAT16, UINT64, UINT32, or UINT16.
  <!-- end id8 -->

- **Return value**:

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
      <td>Returned when self, other, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>Returned when the data type of self, other, or out is not supported.</td>
    </tr>
    <tr>
      <td>The data types of self and other do not comply with the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The shapes of self and out are different.</td>
    </tr>
    <tr>
      <td>The dimensions of self and out are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceNeScalar

- **Parameters**:

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
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnNeScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceNeScalarGetWorkspaceSize

- **Parameters**:

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>selfRef in the formula.</td>
      <td>Non-contiguous tensor is supported.</td>
      <td>DOUBLE, FLOAT16, BFLOAT16, FLOAT, INT32, INT64, UINT64, INT8, UINT8, INT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclScalar*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT16, BFLOAT16, FLOAT, INT32, INT64, UINT64, INT8, UINT8, INT16, BOOL, COMPLEX64, COMPLEX128</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: UINT64 is not supported.
  <!-- end id9 -->
  <!-- npu="910" id10 -->
  - <term>Atlas training products</term>: BFLOAT16 and UINT64 are not supported.
  <!-- end id10 -->

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 300px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>The input selfRef or other is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of selfRef and other are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of selfRef and other do not comply with the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The dimension of selfRef is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceNeScalar

- **Parameters**

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
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnInplaceNeScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
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

## Constraints

- **Deterministic computation**:
  - aclnnNeScalar&aclnnInplaceNeScalar default to a deterministic implementation.

## Example

The sample code is for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**Example code for calling the aclnnNeScalar&aclnnInplaceNeScalar API:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ne_scalar.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // (boilerplate) Resource initialization
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create the aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Initialize the device and stream. See the ACL API reference.
    // Enter the device ID based on your actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. You need to custom-construct them according to the operator API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void *selfDeviceAddr = nullptr;
    void *otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclScalar *other = nullptr;
    aclTensor *out = nullptr;
    std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    double otherValue = 1.0f;

    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclScalar
    other = aclCreateScalar(&otherValue, aclDataType::ACL_DOUBLE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling aclnnNeScalar
    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API aclnnNeScalarGetWorkspaceSize
    ret = aclnnNeScalarGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory for the workspace based on the workspaceSize calculated by the first-phase API
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnNeScalar
    ret = aclnnNeScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeScalar failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from the device to the host. Modify it based on the operator API definition.
    auto size = GetShapeSize(selfShape);
    std::vector<double> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // Example of calling aclnnInplaceNeScalar
    // 3. Call the CANN operator library API. Replace it with the actual API name.
    LOG_PRINT("\ntest aclnnInplaceNeScalar\n");
    // Call the first-phase API aclnnInplaceNeScalarGetWorkspaceSize
    ret = aclnnInplaceNeScalarGetWorkspaceSize(self, other, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory for the workspace based on the workspaceSize calculated by the first-phase API
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnInplaceNeScalar
    ret = aclnnInplaceNeScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeScalar failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from the device to the host. Modify it based on the operator API definition.
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        selfDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy the aclTensor. Modify it based on the operator API definition.
    aclDestroyTensor(self);
    aclDestroyScalar(other);
    aclDestroyTensor(out);

    // 7. Release the device resources. Modify it based on the operator API definition.
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
