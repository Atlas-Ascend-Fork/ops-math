# aclnnExpSegsumBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:59:22.188Z -->

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

- Description: Backward computation of [aclnnExpSegsum](../../segsum/docs/aclnnExpSegsum_en.md).
- Formula (using 5D input as an example):

  1. Multiply the input gradOutput (N1, N2, N3, N4, N4) by the input gradSelf (the forward output).

     $$
     out\_mul = gradOutput * gradSelf
     $$
  2. Generate a triangular matrix A of shape (N4, N4) with bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  3. Fill the elements in the input $out\_mul$ that correspond to positions where the value in matrix A is True with 0.

     $$
     out\_mul_i=
     \begin{cases}out\_mul_i,\quad A_i==False
     \\0, \quad A_i==True
     \end{cases}
     $$
  4. Reverse the penultimate dimension of out_mul to generate out_flip.
  5. Perform cumulative sum along the penultimate dimension of $out\_flip$. From a dimensional perspective, for an element (with other dimension indices unchanged and the current dimension index increasing sequentially), $out\_cumsum\_{i}$ is the element at the corresponding position in the output tensor.

     $$
     out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
     $$
  6. Reverse the -2 dimension of $out\_cumsum$ to generate out_flip2.
  7. Generate a triangular matrix B of shape (N4, N4) with bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  8. Fill the elements in $out\_flip2$ corresponding to the positions where the value is True in matrix B with 0.

     $$
     out\_flip2_i=
     \begin{cases}out\_flip2_i,\quad B_i==False
     \\0, \quad B_i==True
     \end{cases}
     $$
  9. Return gradInput as the sum of each row in the last dimension of out\_flip2.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnExpSegsumBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call "aclnnExpSegsumBackward" to execute the computation.

```Cpp
aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     gradSelf,
  aclTensor           *gradInput,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnExpSegsumBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnExpSegsumBackwardGetWorkspaceSize

- **Parameters**:

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Represents the gradient for backward computation. Corresponds to `gradOutput` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The last two dimensions of shape have the same size.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradSelf</td>
      <td>Input</td>
      <td>Represents the output of the forward computation. Corresponds to `gradSelf` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape and data type are the same as those of the input `gradOutput`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Input</td>
      <td>Represents the output of the backward computation. Corresponds to `gradInput` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as that of the input `gradOutput`.</li><li>The output dimension must be one less than the input dimension. 3D or 4D is supported. When the input `gradOutput` is 4D, the output dimension size is the same as the first 3 dimensions of `gradOutput`. When the input `gradOutput` is 5D, the output dimension size is the same as the first 4 dimensions of `gradOutput`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
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
      <td>Returns an op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, gradSelf, and gradInput are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of gradOutput, gradSelf, and gradInput are out of the supported range.</td>
    </tr>
    <tr>
      <td>The shapes and data types of gradOutput, gradSelf, and gradInput do not meet parameter requirements.</td>
    </tr>
  </tbody></table>

## aclnnExpSegsumBackward

- **Parameters:**

  <table style="table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnExpSegsumBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnExpSegsumBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_exp_segsum_backward.h"

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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
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
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Construct them as required by the API.
    void* gradOutputDeviceAddr = nullptr;
    void* gradSelfDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* gradSelf = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 1, 4, 4};
    std::vector<int64_t> gradSelfShape = {1, 1, 1, 4, 4};
    std::vector<int64_t> gradInputShape = {1, 1, 1, 4};
    std::vector<float> gradOutputHostData(16, 1);
    std::vector<float> gradSelfHostData(16, 1);
    std::vector<float> gradInputHostData(4, 0);

    // Create the input aclTensor.
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradSelfHostData, gradSelfShape, &gradSelfDeviceAddr, aclDataType::ACL_FLOAT, &gradSelf);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnExpSegsumBackward first-phase API.
    ret = aclnnExpSegsumBackwardGetWorkspaceSize(gradOutput, gradSelf, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the aclnnExpSegsumBackward second-phase API.
    ret = aclnnExpSegsumBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to end.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device-side memory to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor. Modify it according to the specific API definition.
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradSelf);
    aclDestroyTensor(gradInput);

    // 7. Release device resources. Modify it according to the specific API definition.
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradSelfDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
