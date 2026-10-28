# aclnnExpSegsum

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:02:26.266Z -->

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

- Description: Performs segment sum computation. Generates a half-separable matrix where the diagonal is 1 and the upper triangle is 0.
- Formula (using 4D input as an example):

  1. The input self is expanded from (N1, N2, N3, N4) to (N1, N2, N3, N4, 1).
  2. Broadcasting is performed to obtain (N1, N2, N3, N4, N4).
  3. Generate a triangular matrix A of shape (N4, N4) with bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is True.
  4. Fill the elements in self that correspond to positions where the value in matrix A is True with 0.

      $$
      self_i=
      \begin{cases}self_i,\quad A_i==False
      \\0, \quad A_i==True
      \end{cases}
      $$

  5. Perform cumsum accumulation along the second-to-last dimension of self. For an element from the dimensional perspective (with other dimension indices unchanged and the current dimension index increasing sequentially), $selfTemp_{i}$ is the element at the corresponding position in the output tensor.

      $$
      selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
      $$

  6. Generate a triangular matrix B of shape (N4, N4) with bool type, where the upper triangle is True, the lower triangle is False, and the diagonal is False.
  7. Fill the elements in selfTemp that correspond to positions where the value in matrix B is True with -inf.

     $$
     out_i=
     \begin{cases}selfTemp_i,\quad B_i==False
     \\-inf, \quad B_i==True
     \end{cases}
     $$

  8. Calculate the exponential of each element in selfTemp.

     $$
     out_i=e^{selfTemp_i}
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnExpSegsumGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the computation process. Then call "aclnnExpSegsum" to perform the computation.

```Cpp
aclnnStatus aclnnExpSegsumGetWorkspaceSize(
  const aclTensor   *self,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnExpSegsum(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnExpSegsumGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input for the segment sum computation, corresponding to `self` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When the tail axis is too large, the output occupies excessive space. For example, when the tail axis of the input is N, the output memory usage is N times that of the input.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output of the segment sum computation, corresponding to `out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as the data type of input `self`.</li><li>The output must have 1 more dimension than the input.<ul><li>If input `self` is 3D, the dimension of the first 3 dimensions of the output is the same as that of `self`, and the dimension of the last dimension is the same as that of the 3rd dimension.</li><li>If input `self` is 4D, the dimension of the first 4 dimensions of the output is the same as that of `self`, and the dimension of the last dimension is the same as that of the 4th dimension.</li></ul></li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the Device-side-applied workspace size.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
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
      <td>self or out is a null pointer, or workspaceSize or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of self and out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self and out do not meet parameter requirements.</td>
    </tr>
    <tr>
      <td>The format of self or out is not ND.</td>
    </tr>
  </tbody></table>

## aclnnExpSegsum

- **Parameters**:

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
      <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size side-applied on the Device, obtained by the first-phase API aclnnExpSegsumGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that specifies the task to be executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnExpSegsum defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_segsum.h"

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
    // Call aclrtMalloc to apply for memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the memory on the device side.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> selfShape = {1, 1, 1, 4};
    std::vector<int64_t> outShape = {1, 1, 1, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3};
    std::vector<int64_t> outputSizeHostData = {16};
    std::vector<float> outHostData(16, 0);

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library APIs. Modify them to the specific APIs.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API aclnnExpSegsum.
    ret = aclnnExpSegsumGetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API aclnnExpSegsum.
    ret = aclnnExpSegsum(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnExpSegsum failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values, and copy the result from device memory to host memory. Modify based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor. Modify based on the definition of the specific API.
    aclDestroyTensor(self);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify based on the definition of the specific API.
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
