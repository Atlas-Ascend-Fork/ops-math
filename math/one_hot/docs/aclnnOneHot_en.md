# aclnnOneHot

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:02:31.894Z -->

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

- Description: For an input self of length n, the one_hot computation produces an output out with n*k elements, where the value of k is numClasses.

- Formula:

  $$
  out[i][j]=\left\{
  \begin{aligned}
  onValue,\quad self[i] = j \\
  offValue, \quad self[i] \neq j
  \end{aligned}
  \right.
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnOneHotGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnOneHot" API to perform the computation.

```Cpp
aclnnStatus aclnnOneHotGetWorkspaceSize(
  const aclTensor* self,
  int              numClasses,
  const aclTensor* onValue,
  const aclTensor* offValue,
  int64_t          axis,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnOneHot(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnOneHotGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 320px">
  <col style="width: 300px">
  <col style="width: 120px">
  <col style="width: 130px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>-</td>
      <td>UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numClasses (int)</td>
      <td>Input</td>
      <td>Indicates the number of classes.</td>
      <td><ul><li>When self is an empty tensor, numClasses must be greater than 0;</li><li>When self is not an empty tensor, numClasses must be greater than or equal to 0.</li><li>If numClasses is 0, an empty tensor is returned.</li><li>If self contains elements greater than numClasses, these elements are encoded entirely as offValue.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>onValue (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the fill value for index positions, that is, onValue in the formula.</td>
      <td><ul><li>Only the first element value is used for computation.</li><li>The data type must be the same as that of out.</li></ul></td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offValue (aclTensor*)</td>
      <td>Input</td>
      <td>Fill value for non-index positions, corresponding to offValue in the formula.</td>
      <td><ul><li>Only the first element value is used for computation.</li><li>The data type is the same as that of out.</li></ul></td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Specifies the dimension where the encoding vector is inserted.</td>
      <td><ul><li>The minimum value is -1, and the maximum value is the number of dimensions of self.</li><li>If the value is -1, the encoding vector is inserted into the last dimension of self.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the one-hot tensor, which is the output out in the formula.</td>
      <td>The output dimension is one more than that of self. The output shape is the same as the shape of self with numClasses inserted along the axis dimension.</td>
      <td>FLOAT16, FLOAT, INT8, UINT8, INT32, INT64</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
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
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: INT8 and UINT8 data types are not supported.
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self, onValue, offValue, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self, onValue, offValue, or out is not within the range of supported data types.</td>
    </tr>
    <tr>
      <td>The data type of onValue or offValue is inconsistent with the data type of out.</td>
    </tr>
    <tr>
      <td>self is an empty tensor and numClasses is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>self is not an empty tensor and numClasses is less than 0.</td>
    </tr>
    <tr>
      <td>The value of axis is less than -1.</td>
    </tr>
    <tr>
      <td>The value of axis is greater than the number of dimensions of self.</td>
    </tr>
    <tr>
      <td>The number of dimensions of out is not 1 greater than that of self.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the shape of self after numClasses is inserted along the axis dimension.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self, onValue, offValue, or out exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnOneHot

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
        <td>Workspace memory address requested on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnOneHotGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation flow.</td>
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
  - aclnnOneHot defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_one_hot.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: resource initialization
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

    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // 1. (boilerplate) Initialize device/stream. For details, see the ACL API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Construct them based on the API as required.
    std::vector<int64_t> selfShape = {4, 2};
    int numClasses = 4;
    std::vector<int64_t> outShape = {4, 2, 4};
    std::vector<int64_t> onValueShape = {1};
    std::vector<int64_t> offValueShape = {1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* onValueDeviceAddr = nullptr;
    void* offValueDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* onValue = nullptr;
    aclTensor* offValue = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int32_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> onValueHostData = {1};
    std::vector<int32_t> offValueHostData = {0};
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the onValue aclTensor.
    ret = CreateAclTensor(onValueHostData, onValueShape, &onValueDeviceAddr, aclDataType::ACL_INT32, &onValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offValue aclTensor.
    ret = CreateAclTensor(offValueHostData, offValueShape, &offValueDeviceAddr, aclDataType::ACL_INT32, &offValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    int64_t axis = -1;
    aclOpExecutor* executor;
    // Call the aclnnOneHot first-phase API.
    ret = aclnnOneHotGetWorkspaceSize(self, numClasses, onValue, offValue, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHotGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the aclnnOneHot second-phase API.
    ret = aclnnOneHot(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnOneHot failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result in device memory to the host. Modify it based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the definition of the specific API.
    aclDestroyTensor(self);
    aclDestroyTensor(onValue);
    aclDestroyTensor(offValue);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify it based on the definition of the specific API.
    aclrtFree(selfDeviceAddr);
    aclrtFree(onValueDeviceAddr);
    aclrtFree(offValueDeviceAddr);
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
