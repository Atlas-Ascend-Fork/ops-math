# aclnnArgsort

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:15:05.255Z -->

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

- Description: Sorts the elements in the input tensor along a specified dimension in ascending or descending order and returns the corresponding indices.
- Formula:

$$
out[\mathbf{i}_{\neg dim}, j] = \text{arg\_sort}_{dim}(self[\mathbf{i}_{\neg dim}, :])[j]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnArgsortGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call "aclnnArgsort" to perform the computation.

```Cpp
  aclnnStatus aclnnArgsortGetWorkspaceSize(
  const aclTensor *self,
  int64_t          dim,
  bool             descending,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
  aclnnStatus aclnnArgsort(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnArgsortGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1499px"><colgroup>
  <col style="width: 165px">
  <col style="width: 125px">
  <col style="width: 294px">
  <col style="width: 247px">
  <col style="width: 261px">
  <col style="width: 124px">
  <col style="width: 138px">
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
      <th>Dimension(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>Supports empty tensors.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>The dimension used as the sorting criterion.</td>
      <td>The range is [-self.dim(), self.dim()-1].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>Input</td>
      <td>Controls the sort order. True for descending order, False for ascending order.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Supports empty tensors.</td>
      <td>INT64</td>
      <td>Same as self</td>
      <td>Same as self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The data types UINT16, UINT32, and UINT64 are not supported.
    - When the data type of self is BFLOAT16, the sort axis 1 is not supported.
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
      <td>self and out are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or data format of self and out is not within the supported range, or the shapes do not match.</td>
    </tr>
    <tr>
      <td>The value of dim is not in the range [-N, N-1].</td>
    </tr>
    <tr>
      <td>The data type of self is BFLOAT16 and the sorting axis is 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnArgsort

- **Parameters:**

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the device, obtained by calling the first-phase API aclnnArgsortGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
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

<br />

## Constraints

- Deterministic computation:
  - aclnnArgsort defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_argsort.h"

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
    // Call aclrtMemcpy to copy host data to device memory.
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
    // 1. (Boilerplate) Initialize device/stream. Refer to the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct input and output. The construction must be customized based on the API.
    int64_t dim = 0;
    bool descending = false;
    std::vector<int64_t> selfShape = {3, 4};
    std::vector<int64_t> outIndicesShape = {3, 4};
    void* selfDeviceAddr = nullptr;
    void* outIndicesDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* outIndices = nullptr;
    std::vector<int64_t> selfHostData = {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
    std::vector<int64_t> outIndicesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the outIndices aclTensor.
    ret = CreateAclTensor(outIndicesHostData, outIndicesShape, &outIndicesDeviceAddr, aclDataType::ACL_INT64, &outIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnArgsort.
    ret = aclnnArgsortGetWorkspaceSize(self, dim, descending, outIndices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgsortGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnArgsort.
    ret = aclnnArgsort(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgsort failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for task execution to finish synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device to the host. Modify it based on the specific API definition.
    auto size2 = GetShapeSize(outIndicesShape);
    std::vector<int64_t> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), outIndicesDeviceAddr,
                      size2 * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("result indices [%ld] is: %ld\n", i, resultData2[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(outIndices);

     // 7. Release device resources.
    aclrtFree(selfDeviceAddr);
    aclrtFree(outIndicesDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
