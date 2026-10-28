# aclnnAddN

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:09:05.895Z -->

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
- <term>Atlas training products</term>: not supported
<!-- end id6 -->

## Function

- Description: Adds the tensors in the input TensorList element by element to compute the sum.

- Formula:

  $$
  out = tensors_{1} + tensors_{2} + \dots + tensors_{n}
  $$

## Function Prototype

Each operator is a two-phase API. You must call the "aclnnAddNGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnAddN" API to perform computation.

```cpp
aclnnStatus aclnnAddNGetWorkspaceSize(
  const aclTensorList *tensors,
  aclTensor           *out,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnAddN(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnAddNGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>tensors (const aclTensorList*)</td>
      <td>Input</td>
      <td>Input TensorList. The operator performs element-wise summation on the input tensors.</td>
      <td><li>An empty TensorList is supported. In this case, the output is an empty Tensor.</li><li>The Tensors in tensors must have consistent shapes.</li></td>
      <td>INT32, INT64, FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output Tensor. Stores the summation result.</td>
      <td><li>Empty tensors are not supported.</li><li>The data type must be consistent with that of the tensors in tensors.</li><li>The shape must be consistent with the shape of the tensors in tensors.</li></td>
      <td>The data type is consistent with that of tensors.</td>
      <td>ND</td>
      <td>1-8</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
    <td>tensors is a null pointer, or a Tensor in tensors is a null pointer, or out is a null pointer.</td>
  </tr>
  <tr>
    <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="5">161002</td>
    <td>The data type of tensors or out is not within the supported range.</td>
  </tr>
  <tr>
    <td>The number of shape dimensions of tensors or out exceeds 8.</td>
  </tr>
  <tr>
    <td>The data types of tensors in tensors are not consistent.</td>
  </tr>
  <tr>
    <td>The shapes of tensors in tensors are not consistent.</td>
  </tr>
  <tr>
    <td>The operator runs on an unsupported product model.</td>
  </tr>
  </tbody></table>

## aclnnAddN

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
          <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
      </thead>
      <tbody>
          <tr><td>workspace</td><td>Input</td><td>Workspace memory address requested on the Device side.</td></tr>
          <tr><td>workspaceSize</td><td>Input</td><td>Workspace size requested on the Device side, obtained by the first-phase API aclnnAddNGetWorkspaceSize.</td></tr>
          <tr><td>executor</td><td>Input</td><td>Op executor that contains the operator computation process.</td></tr>
          <tr><td>stream</td><td>Input</td><td>Stream for task execution.</td></tr>
      </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnAddN defaults to a deterministic implementation.

<!-- npu="950,A3,910b" id7 -->
- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - Data types: INT32, INT64, FLOAT16, BFLOAT16, and FLOAT32 are supported.
  - Data format: only the ND format is supported.
  - Parameter shape: the dimension range of input tensors is 1-8, and shapes must be consistent.
<!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_n.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of contiguous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    int32_t deviceId = 0; // Set deviceId based on the actual device.
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API definition.
    std::vector<int64_t> inputShape1 = {4, 2};
    std::vector<int64_t> inputShape2 = {4, 2};
    std::vector<int64_t> outputShape = {4, 2};
    void* inputDeviceAddr1 = nullptr;
    void* inputDeviceAddr2 = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* output = nullptr;
    std::vector<float> inputHostData1 = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> inputHostData2 = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<float> outputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    aclDataType dtype = aclDataType::ACL_FLOAT;

    // Create the aclTensor of tensors.
    ret = CreateAclTensor(inputHostData1, inputShape1, &inputDeviceAddr1, dtype, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData2, inputShape2, &inputDeviceAddr2, dtype, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the aclTensor of out.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, dtype, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the aclTensorList.
    std::vector<aclTensor*> tmp{ input1, input2 };
    aclTensorList* inputList = aclCreateTensorList(tmp.data(), tmp.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the first-phase API aclnnAddNGetWorkspaceSize.
    ret = aclnnAddNGetWorkspaceSize(inputList, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory using the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnAddN.
    ret = aclnnAddN(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddN failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value. Copy the data on the device side to the host side. Modify the code based on the specific API definition.
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclTensorList. Modify the code based on the specific API definition.
    aclDestroyTensor(output);
    aclDestroyTensorList(inputList);

    // 7. Release device resources
    aclrtFree(inputDeviceAddr1);
    aclrtFree(inputDeviceAddr2);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
