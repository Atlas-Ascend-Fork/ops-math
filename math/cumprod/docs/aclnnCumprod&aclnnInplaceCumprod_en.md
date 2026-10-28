# aclnnCumprod&aclnnInplaceCumprod

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:47:25.768Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: The `cumprod` function computes the cumulative product of the input tensor along a specified dimension. For example, given a tensor representing a sequence of values, `cumprod` can compute the product sequence from the starting position to the current position.

- Formula:

  - **One-Dimensional Tensor (Vector) Case**
      For a one-dimensional tensor, the cumulative product $y=[y_1,y_2,y_3...,y_n]$ is calculated as:

      $y_1=x_1$
      $y_2=x_1 \times x_2$
      $y_3=x_1 \times x_2\times x_3$
      ...
      $y_n=x_1\times x_2\times x_3\times x_n$

      Expressed in mathematical formula $y_i=\prod_{j=1}^ix_j, where i=1,2...,n$.

  - **High-dimensional tensor case (taking a two-dimensional tensor as an example, with dim=0 along the row direction)**
    For a two-dimensional tensor:

    $$
    X=\begin{bmatrix}x_{11}&x_{12}&...&x_{1m}\\x_{21}&x_{22}&...&x_{2m}\\...&...&...&...&\\x_{n1}&x_{n2}&...&x_{nm}&\end{bmatrix}
    $$

    The resulting tensor after computation:

    $$
      Y=\begin{bmatrix}y_{11}&y_{12}&...&y_{1m}\\y_{21}&y_{22}&...&y_{2m}\\...&...&...&...&\\y_{n1}&y_{n2}&...&y_{nm}&\end{bmatrix}
    $$

    For the first column (j=1):

    $
    y_{i1}=x_{11}\times x_{21}\times ...\times x_{i1}(\text{for }i=1,2,....n)
    $

    Therefore, for any column j, a similar pattern also applies:

    $$
    y_{ij}=\prod_{k=1}^{i} x_{kj}
    $$

  - **High-dimensional tensor (taking a two-dimensional tensor as an example, with dim=1 along the column direction)**
    Therefore, for any column j, a similar pattern also applies:

    $$
    y_{ij}=\prod_{k=1}^{j} x_{ik}
    $$

  - **Other parameters can be derived similarly based on the rules above.**

## Function Prototype

aclnnCumprod and aclnnInplaceCumprod implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

- aclnnCumprod: Create a new output tensor object to store the computation result.
- aclnnInplaceCumprod: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnCumprodGetWorkspaceSize" or "aclnnInplaceCumprodGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnCumprod" or "aclnnInplaceCumprod" API to perform the computation.

```Cpp
aclnnStatus aclnnCumprodGetWorkspaceSize(
  const aclTensor*   input,
  const aclScalar*   dim,
  const aclDataType  dtype,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnCumprod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

```Cpp
aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(
  aclTensor*       input,
  const aclScalar* dim,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceCumprod(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnCumprodGetWorkspaceSize

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
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Data to compute the cumulative product.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>Up to 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclScalar*)</td>
      <td>Input</td>
      <td>Dimension along which the cumulative product is computed. For a 2D tensor, dim=0 indicates computation along the row direction, and dim=1 indicates computation along the column direction.</td>
      <td>Value range: [-rank(input), rank(input)).</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype (aclDataType)</td>
      <td>Input</td>
      <td>Specifies the data type of input for the specified computation process.</td>
      <td><ul><li>If the value is ACL_DT_UNDEFINED, the computation uses the original data type of the passed input.</li><li>If a specific data type is specified (it must be within the data types supported by input), input is converted to this data type before computation.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Cumulative product result.</td>
      <td><ul><li>If dtype=ACL_DT_UNDEFINED, the data type must be the same as that of input.</li><li>If dtype is specified, the data type must be the same as that specified by dtype.</li><li>The shape of out must be the same as that of input.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor. It contains the operator computation flow.</td>
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
      <td>The input and dim are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and format of input and dim are out of the supported range.</td>
    </tr>
    <tr>
      <td>The shape constraint between dim and input is not met.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with that of input.</td>
    </tr>
  </tbody>
  </table>

## aclnnCumprod

- **Parameters:**

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnCumprodGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
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

## aclnnInplaceCumprodGetWorkspaceSize

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
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Data and result for computing the cumulative product.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim (aclScalar*)</td>
      <td>Input</td>
      <td>Specifies the dimension for computing the cumulative product. For a two-dimensional tensor, dim=0 indicates computing along the row direction, and dim=1 indicates computing along the column direction.</td>
      <td>Value range: [-rank(input), rank(input)].</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied for on the device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input or dim is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type and format of the passed-in input and dim are not supported.</td>
    </tr>
    <tr>
      <td>The shape constraints between the passed-in dim and input are not met.</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceCumprod

- **Parameters:**

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnInplaceCumprodGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. It contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnCumprod and aclnnInplaceCumprod default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumprod.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++)
    {
        LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
    }
}

template<typename T>
void PrintOutFloatResult(std::vector<T> &shape, void **deviceAddr, const char *name)
{
    std::vector<float> resultData(shape.size(), 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, shape.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < shape.size(); i++)
    {
        LOG_PRINT("result var %s[%ld] is: %f\n", name, i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Boilerplate for ACL initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclScalar(aclDataType dataType, T &hostData, aclScalar **scalar)
{
    *scalar = aclCreateScalar(&hostData, dataType);
    if (*scalar == nullptr)
    {
        return -1;
    }
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the ACL external API list. Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Perform the custom construction based on the API.
    void *xDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    std::vector<int64_t> xShape = {3};
    std::vector<int64_t> xHostData = {1,2,3};
    // Create the original input x.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT64, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the axis aclScalar.
    int32_t axis_value = 0;
    aclScalar *axis = nullptr;
    ret = CreateAclScalar(aclDataType::ACL_INT32, axis_value, &axis);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the result aclTensor.
    std::vector<int64_t> resultHostData(3, 0);
    std::vector<int64_t> resultShape = {3};
    void *resultDeviceAddr = nullptr;
    aclTensor *result = nullptr;
    ret = CreateAclTensor(resultHostData, resultShape, &resultDeviceAddr, aclDataType::ACL_INT64, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    aclDataType dtype = ACL_INT64;
    void *workspaceAddr = nullptr;
    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnCumprod.
    ret = aclnnCumprodGetWorkspaceSize(input, axis, dtype, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnCumprod.
    ret = aclnnCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4.(boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    // 5.Obtain the output value and copy the result from device memory to host memory. Modify this step based on the specific API definition.
    PrintOutResult(resultShape, &resultDeviceAddr);

    // 3.Call the CANN operator library APIs.
    // Call the first-phase API of aclnnInplaceCumprod.
    ret = aclnnInplaceCumprodGetWorkspaceSize(input, axis, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCumprodGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCumprod allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnInplaceCumprod.
    ret = aclnnInplaceCumprod(workspaceAddr, workspaceSize, executor, stream);
    // 4.(boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    // 5. Obtain the output value and copy the result on device memory to the host. Modify the code based on the specific API definition.
    PrintOutResult(resultShape, &xDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(input);
    aclDestroyScalar(axis);

    // // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
