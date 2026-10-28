# aclnnNeTensor&aclnnInplaceNeTensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:38:00.303Z -->

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

- Description: Computes whether the value of each element in self (selfRef) is not equal to the value in other.
- Formula:

$$
out_i​=(self_i \ne other_i)?[1]: [0]
$$

$$
selfRef_i​=(selfRef_i\ \ne other_i)\ ?\  [1]: [0]
$$

## Function Prototype

- aclnnNeTensor and aclnnInplaceNeTensor implement the same functionality. The differences are as follows. Choose the appropriate operator based on your actual scenario.
  - aclnnNeTensor: Create a new output tensor object to store the computation result.
  - aclnnInplaceNeTensor: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnNeTensorGetWorkspaceSize" or "aclnnInplaceNeTensorGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the computation process. Then call "aclnnNeTensor" or "aclnnInplaceNeTensor" to perform the computation.

```Cpp
aclnnStatus aclnnNeTensorGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *other,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnNeTensor(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

```Cpp
aclnnStatus aclnnInplaceNeTensorGetWorkspaceSize(
    aclTensor       *selfRef,
    const aclTensor *other,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceNeTensor(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnNeTensorGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>The data type needs to satisfy the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">data type deduction rule</a> with other, and the shape needs to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with other.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>The data type needs to be consistent with other based on the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">data type deduction rules</a>, and the shape needs to be consistent with other based on the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>The data type needs to be a <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">data type</a> convertible to BOOL, and the shape needs to be the broadcast shape of self and other (see the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>).</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64, UINT32, UINT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**:

  aclnnStatus: return status code, see specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>When self, other, and out are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>When the data type of self, other, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>When the dimension of self, other, or out is greater than 8.</td>
    </tr>
    <tr>
      <td>When the data types of self and other cannot be deduced.</td>
    </tr>
    <tr>
      <td>When the shapes of self and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>When the shape of out is inconsistent with the shape after broadcast.</td>
    </tr>
  </tbody>
  </table>

## aclnnNeTensor

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnNeTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the operator computing process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code, see specifically [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceNeTensorGetWorkspaceSize

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
      <td>selfRef(aclTensor*)</td>
      <td>Input/Output</td>
      <td>selfRef in the formula.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other (aclTensor*)</td>
      <td>Input</td>
      <td>other in the formula.</td>
      <td>DOUBLE, FLOAT16, FLOAT, BFLOAT16, INT64, INT32, INT8, UINT8, BOOL, INT16, COMPLEX64, COMPLEX128, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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
      <td>Returns the op executor, which includes the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**:

  aclnnStatus: Return status code. See specifically [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>When selfRef and other are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>When the data type of selfRef or other is not in the supported range.</td>
    </tr>
    <tr>
      <td>When the data type of selfRef or other cannot be inferred.</td>
    </tr>
    <tr>
      <td>When the shapes of selfRef and other cannot be broadcast.</td>
    </tr>
    <tr>
      <td>When the shape of selfRef and other after broadcasting is not equal to the shape of selfRef.</td>
    </tr>
    <tr>
      <td>When the dimension of selfRef or other is greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceNeTensor

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnInplaceNeTensorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which includes the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. See specifically [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNeTensor&aclnnInplaceNeTensor default to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**Example code for calling the aclnnNeTensor&aclnnInplaceNeTensor API:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ne_tensor.h"

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
aclDataType dataType, aclTensor **tensor)
    {
auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the Device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the Host-side data to the Device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

// Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
        }

// Call the aclCreateTensor API to create an aclTensor.
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
// 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. The construction needs to be based on the API and customized.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {4, 2};
    std::vector<int64_t> outShape = {4, 2};
    void *selfDeviceAddr = nullptr;
    void *otherDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *other = nullptr;
    aclTensor *out = nullptr;
    std::vector<double> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<double> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<double> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

// Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_DOUBLE, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclTensor.
    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_DOUBLE, &other);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_DOUBLE, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

// aclnnNeTensor API call example.
    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnNeTensor first-phase API.
    ret = aclnnNeTensorGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
    // Call the aclnnNeTensor second-phase API.
    ret = aclnnNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeTensor failed. ERROR: %d\n", ret); return ret);

// 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. Obtain the output value and copy the result on the Device side memory to the Host side.
    auto size = GetShapeSize(outShape);
    std::vector<double> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
                      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
        }

//aclnnInplaceNeTensor API call example.
    // 3. Call the CANN operator library API.
    LOG_PRINT("\ntest aclnnInplaceNeTensor\n");
    // For the inplace API, selfRef is the input and output tensor (self is updated in place). Use selfRef as the input parameter.
    aclTensor *selfRef = self;
    // Call the aclnnInplaceNeTensor first-phase API.
    ret = aclnnInplaceNeTensorGetWorkspaceSize(selfRef, other, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API. Release the workspace applied for by the first-phase API first to avoid memory leaks.
    if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
        workspaceAddr = nullptr;
        }
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
    // Call the aclnnInplaceNeTensor second-phase API.
    ret = aclnnInplaceNeTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceNeTensor failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. Obtain the output value and copy the result on the device side memory to the host side. The code needs to be modified based on the interface definition of the specific API.
    ret = aclrtMemcpy(resultData.data(),
    resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
        }

// 6. Release the aclTensor. The code needs to be modified based on the interface definition of the specific API.
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);

// 7. Release device resources. The code needs to be modified based on the interface definition of the specific API.
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
