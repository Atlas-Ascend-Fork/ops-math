# aclnnAny

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:30:37.681Z -->

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

For each dimension in the given `dim`, if any element corresponding to that dimension in the input tensor evaluates to True, returns True; otherwise, returns False. If `keepdim` is True, the output tensor has the same number of dimensions as the input; otherwise, the `dim` dimensions are squeezed, resulting in the output tensor having `len(dim)` fewer dimensions.

For example: if the input tensor has a shape of $(A\times B \times C \times D)$ and `dim` is [0, 2], the output tensor has a shape of $(B \times D)$, with two fewer dimensions than the input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnAnyGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnAny" API to perform the computation.

```Cpp
aclnnStatus aclnnAnyGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *dim,
  bool               keepdim,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnAny(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnAnyGetWorkspaceSize

- **Parameters:**

    <table style="table-layout: fixed; width: 1494px"><colgroup>
    <col style="width: 154px">
    <col style="width: 125px">
    <col style="width: 259px">
    <col style="width: 209px">
    <col style="width: 334px">
    <col style="width: 124px">
    <col style="width: 138px">
    <col style="width: 151px">
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
        <td>self</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>-</td>
        <td>BOOL, INT8, UINT8, INT16, INT32, INT64, BFLOAT16, FLOAT16, FLOAT32, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Dimension to reduce.</td>
        <td>The value must be within the range of the input tensor dimensions. Negative values are supported. Range: [-self.dim(), self.dim() - 1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepdim</td>
        <td>Input</td>
        <td>Whether the dimension of the reduced axis is kept.</td>
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
        <td>-</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the Device-side-requested workspace.</td>
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

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas training products</term>, <term>Atlas 200I/500 A2 inference products</term>: do not support the BFLOAT16 data type.
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
        <th>Error code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>self, out, or dim is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The data type of self or out is not in the supported range.</td>
      </tr>
      <tr>
        <td>The dimension specified by dim is out of the valid range.</td>
      </tr>
    </tbody>
    </table>

## aclnnAny

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
        <td>Device side-requested workspace memory address.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Device side-requested workspace size, obtained by the first-phase API aclnnAnyGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op. It contains the operator computation flow.</td>
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
  - aclnnAny defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_any.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate code for resource initialization.
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

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. Boilerplate for device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definitions.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {1, 2};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray * dim = nullptr;
    aclTensor* out = nullptr;

    std::vector<int> selfHostData = {0, 1, 0, 3, 0, 5, 0, 7};
    std::vector<unsigned char> outHostData = {0, 0};
    std::vector<int64_t> dimData = {0};
    bool keepdim = true;

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the dim aclIntArray.
    dim = aclCreateIntArray(dimData.data(), 1);
    CHECK_RET(dim != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_BAD_ALLOC);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BOOL, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAny.
    ret = aclnnAnyGetWorkspaceSize(self, dim, keepdim, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAnyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAny.
    ret = aclnnAny(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAny failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value. Copy the result from the device side to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<unsigned char> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(unsigned char),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %hhu\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify them according to the specific API definition.
    aclDestroyTensor(self);
    aclDestroyIntArray(dim);
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
