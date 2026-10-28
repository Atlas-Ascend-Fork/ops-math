# aclnnLeftShift

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:20:59.714Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: For each element in the input tensor self, performs a bitwise left shift based on the corresponding element in the input tensor shiftBits.
- Formula:

$$
out_{i} = self_{i}<<shiftBits_{i}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnLeftShiftGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnLeftShift" to perform the computation.

```c++
aclnnStatus aclnnLeftShiftGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *shiftBits,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnLeftShift(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnLeftShiftGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1542px"><colgroup>
  <col style="width: 141px">
  <col style="width: 120px">
  <col style="width: 344px">
  <col style="width: 431px">
  <col style="width: 161px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <th>Dimension</th>
      <th>non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be bitwise left shifted, self in the formula.</td>
      <td>Empty tensors are supported.<br>The data type must satisfy the data type deduction rules together with the data type of shiftBits (see <a href="../../../docs/en/context/deduction_relationship.md" class="md-link">deduction relationship</a>).<br>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" class="md-link">broadcast relationship</a> with shiftBits.</td>
      <td>INT8, INT16, INT32, INT64,<br>UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shiftBits</td>
      <td>Input</td>
      <td>Tensor of the left shift operand, shiftBits in the formula.</td>
      <td>Empty tensors are supported.<br>The data type must follow the data type deduction rule with self (see <a href="../../../docs/en/context/deduction_relationship.md" class="md-link">deduction relationship</a>).<br>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md" class="md-link">broadcast relationship</a> with self.</td>
      <td>INT8, INT16, INT32, INT64, <br>UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is out in the formula.</td>
      <td>The shape must be the same as the shape after broadcasting self and shiftBits.</td>
      <td>INT8, INT16, INT32, INT64, <br>UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 123px">
  <col style="width: 741px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, shiftBits, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types and data formats of self, shiftBits, and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and shiftBits do not support type deduction.</td>
    </tr>
    <tr>
      <td>The dimensions of self and shiftBits are greater than 8.</td>
    </tr>
    <tr>
      <td>The shapes of self and shiftBits do not meet the broadcast rules.</td>
    </tr>
    <tr>
      <td>The shapes of self and shiftBits are inconsistent with the shape of out after broadcasting.</td>
    </tr>
  </tbody>
  </table>

## aclnnLeftShift

- **Parameters:**

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnLeftShiftGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor. It contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLeftShift defaults to a deterministic implementation.

- Accuracy is guaranteed only when the value of shiftBits is less than the bit width of the self data type.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_left_shift.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data on the host to memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Device/stream initialization. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Perform custom construction as required by the API.
    std::vector<int64_t> selfShape = {1, 4};
    std::vector<int64_t> shiftBitsShape = {1, 4};
    std::vector<int64_t> outShape = {1, 4};
    void* selfDeviceAddr = nullptr;
    void* shiftBitsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* shiftBits = nullptr;
    aclTensor* out = nullptr;
    std::vector<int64_t> selfHostData = {10, 20, 30, 40};
    std::vector<int64_t> shiftBitsHostData = {1, 2, 3, 4};
    std::vector<int64_t> outHostData = {0, 0, 0, 0};

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the shiftBits aclTensor.
    ret = CreateAclTensor(shiftBitsHostData, shiftBitsShape, &shiftBitsDeviceAddr, aclDataType::ACL_INT64, &shiftBits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnLeftShiftGetWorkspaceSize first-phase API.
    ret = aclnnLeftShiftGetWorkspaceSize(self, shiftBits, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeftShiftGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the aclnnLeftShift second-phase API.
    ret = aclnnLeftShift(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLeftShift failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device memory to the host. Modify it based on the actual API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int64_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the actual API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(shiftBits);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(selfDeviceAddr);
    aclrtFree(shiftBitsDeviceAddr);
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
