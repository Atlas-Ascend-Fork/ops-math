# aclnnAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:29:55.348Z -->

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

- Description: For each dimension in the given dimension `dim`, if all elements corresponding to that dimension in the input tensor evaluate to True, returns True; otherwise, returns False. If `keepdim` is True, the output tensor has the same number of dimensions as the input; otherwise, the `dim` dimensions are squeezed, resulting in the output tensor having `len(dim)` fewer dimensions.
- For example, if the input tensor has the shape $(A\times B \times C \times D)$ and `dim` is [0, 2], then if `keepdim` is False, the output tensor has the shape $(B \times D)$, with two fewer dimensions than the input; if `keepdim` is True, the output tensor has the shape $(1 \times B \times 1 \times D)$, with the same number of dimensions as the input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAllGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the process, and then call the "aclnnAll" API to perform the computation.

```Cpp
aclnnStatus aclnnAllGetWorkspaceSize(
const aclTensor   *self,
const aclIntArray *dim,
bool               keepdim,
aclTensor         *out,
uint64_t          *workspaceSize,
aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnAll(
void             *workspace,
uint64_t          workspaceSize,
aclOpExecutor    *executor,
const aclrtStream stream)
```

## aclnnAllGetWorkspaceSize

- **Parameters:**

    <table style="table-layout: fixed; width: 1499px"><colgroup>
    <col style="width: 153px">
    <col style="width: 120px">
    <col style="width: 269px">
    <col style="width: 277px">
    <col style="width: 283px">
    <col style="width: 119px">
    <col style="width: 132px">
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
        <td>self</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>-</td>
        <td>BOOL, UINT8, INT8, INT16, INT32, INT64, FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Dimension to reduce.</td>
        <td>The value must be within the range of the input tensor. Negative values are supported. Value range: [-self.dim(), self.dim() - 1].<br>If dim is [], reduction is performed over all dimensions.</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepdim</td>
        <td>Input</td>
        <td>Whether to keep the reduced dimension.</td>
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
        <td>Returns the size of the workspace to be side-requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Op executor, containing the operator computation flow.</td>
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

    <table style="table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 290px">
    <col style="width: 114px">
    <col style="width: 706px">
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
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>The data types of self and out are not within the supported range.</td>
      </tr>
      <tr>
        <td>The dimension specified by dim is out of the valid range.</td>
      </tr>
      <tr>
        <td>The dim array contains duplicate elements.</td>
      </tr>
      <tr>
        <td>The shape of out does not match the actual shape.</td>
      </tr>
    </tbody>
    </table>

## aclnnAll

- **Parameters:**

    <table style="table-layout: fixed; width: 1110px"><colgroup>
    <col style="width: 153px">
    <col style="width: 124px">
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
        <td>Device side-requested workspace memory address.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Device side-requested workspace size, obtained by the first-phase API aclnnAllGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAll defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_all.h"
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
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. Boilerplate. Initialize the device/stream. For details, see the acl API manual
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definitions
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> outShape = {1, 2};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray * dim = nullptr;
    aclTensor* out = nullptr;

    std::vector<int> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<unsigned char> outHostData = {0, 0};
    std::vector<int64_t> dimData = {0};
    bool keepdim = true;

    // Create the self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the dim aclIntArray
    dim = aclCreateIntArray(dimData.data(), 1);
    CHECK_RET(dim != nullptr, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT8, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAll.
    ret = aclnnAllGetWorkspaceSize(self, dim, keepdim, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAllGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAll.
    ret = aclnnAll(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAll failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Wait for the task to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<unsigned char> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(unsigned char),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %hhu\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
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
