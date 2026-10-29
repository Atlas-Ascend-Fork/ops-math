# aclnnTril&aclnnInplaceTril

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:46:04.429Z -->

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

- Operator function: Zeros out the upper-right portion of the last two dimensions (counted from left to right by shape) of the input self Tensor along the diagonal. The diagonal parameter can be positive or negative, and defaults to zero. A positive value shifts the main diagonal toward the upper-right direction, and a negative value shifts it toward the lower-left direction.
- Formula: In the following, i denotes the index for iterating over the second-to-last dimension (i is the row index), j denotes the index for iterating over the last dimension (j is the column index), and d denotes diagonal. In the 2D coordinate plot corresponding to (i, j), i+d==j indicates that the element lies on the diagonal.

    $
    On and to the lower-left of the diagonal, i.e., i+d>=j, retain the original value: out_{i, j} = self_{i, j}\\
    To the upper-right of the diagonal, i.e., i+d<j, set to zero (excluding the diagonal): out_{i, j} = 0
    $

- Example:

    $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$,
    The result of tril(self, diagonal=0) is:
    $\begin{bmatrix} [9&0&0] \\ [1&2&0] \\ [3&4&1] \end{bmatrix}$;
    With diagonal set to 1, the result of tril(self, diagonal=1) is:
    $\begin{bmatrix} [9&6&0] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$;
    With diagonal set to -1, the result of tril(self, diagonal=-1) is:
    $\begin{bmatrix} [0&0&0] \\ [1&0&0] \\ [3&4&0] \end{bmatrix}$.

## Function Prototype

- aclnnTril and aclnnInplaceTril implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnTril: Create a new output tensor object to store the computation result.
  - aclnnInplaceTril: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnTrilGetWorkspaceSize" or "aclnnInplaceTrilGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnTril" or "aclnnInplaceTril" to execute the computation.

```Cpp
aclnnStatus aclnnTrilGetWorkspaceSize(
  const aclTensor* self,
  int64_t          diagonal,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTril(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceTrilGetWorkspaceSize(
  const aclTensor* selfRef,
  int64_t          diagonal,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceTril(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnTrilGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 349px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>Represents the target tensor to be converted, corresponding to self in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>diagonal (int64_t)</td>
      <td>Input</td>
      <td>Position of the diagonal.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Input</td>
      <td>Output tensor.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>-</td>
      <td>2-8</td>
      <td>-</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output parameter</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output parameter</td>
      <td>Returns the op executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: COMPLEX32 and COMPLEX64 are supported only on these products.
  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="table-layout: fixed; width: 1154px"><colgroup>
    <col style="width: 230px">
    <col style="width: 127px">
    <col style="width: 851px">
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
        <td>self or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of self and out are not within the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The data formats of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self is greater than 8 or less than 2.</td>
      </tr>
    </tbody>
    </table>

## aclnnTril

- **Parameters**
  <table style="table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnTrilGetWorkspaceSize.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceTrilGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 349px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef (aclTensor*)</td>
      <td>Input</td>
      <td>Target tensor to be converted, corresponding to self in the formula.</td>
      <td>-</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL, BFLOAT16, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>diagonal (int64_t)</td>
      <td>Input</td>
      <td>Position of the diagonal.</td>
      <td>-</td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>output parameter</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>output parameter</td>
      <td>Returns the executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody></table>

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: COMPLEX32 and COMPLEX64 are supported only on these products.
  <!-- end id9 -->
  <!-- npu="910,310p" id10 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id10 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="table-layout: fixed; width: 1154px"><colgroup>
    <col style="width: 230px">
    <col style="width: 127px">
    <col style="width: 851px">
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
        <td>The input selfRef is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type of selfRef is not supported.</td>
      </tr>
      <tr>
        <td>The dimension of self is greater than 8, or less than 2 and not 0.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceTril

- **Parameters**
  <table style="table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 230px">
  <col style="width: 127px">
  <col style="width: 851px">
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
      <td>Memory address of the workspace allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnInplaceTrilGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnTril&aclnnInplaceTril default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_tril.h"
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. Boilerplate for device/stream initialization. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs according to the API definition.
    std::vector<int64_t> selfShape = {3, 3};
    std::vector<int64_t> outShape = {3, 3};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<int> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> outHostData ={1, 2, 3, 4, 5, 6, 7, 8, 9};
    int diagonal = 0;

    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnTril.
    ret = aclnnTrilGetWorkspaceSize(self, diagonal, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTrilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnTril.
    ret = aclnnTril(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTril failed. ERROR: %d\n", ret); return ret);

    uint64_t inplaceWorkspaceSize = 0;
    aclOpExecutor* inplaceExecutor;
    // Call the first-phase API of aclnnInplaceTril.
    ret = aclnnInplaceTrilGetWorkspaceSize(self, diagonal, &inplaceWorkspaceSize, &inplaceExecutor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTrilGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* inplaceWorkspaceAddr = nullptr;
    if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnInplaceTril.
    ret = aclnnInplaceTril(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceTril failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device side to the host side. Change them based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    auto inplaceSize = GetShapeSize(selfShape);
    std::vector<int> inplaceResultData(inplaceSize, 0);
    ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                      inplaceSize * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < inplaceSize; i++) {
      LOG_PRINT("inplaceResult[%ld] is: %d\n", i, inplaceResultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Change them based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);

    // 7. Release device resources, and modify the code based on the definition of the specific API.
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
