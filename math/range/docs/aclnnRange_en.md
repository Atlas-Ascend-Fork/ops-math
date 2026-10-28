# aclnnRange

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:24:13.575Z -->

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

- Description: Generates values from start to end at the interval of step and returns a 1-dimensional tensor of size $\lfloor\frac{end-start}{step}\rfloor+1$. The stride step is the interval between two adjacent values in the tensor.

- Formula:

$$
out_{i+1} = out_i+step
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnRangeGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnRange" API to perform computation.

```Cpp
aclnnStatus aclnnRangeGetWorkspaceSize(
  const aclScalar *start,
  const aclScalar *end,
  const aclScalar *step,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRange(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRangeGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 154px">
  <col style="width: 125px">
  <col style="width: 213px">
  <col style="width: 288px">
  <col style="width: 333px">
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
      <td>start</td>
      <td>Input</td>
      <td>Start position of the range from which values are obtained.</td>
      <td>The input start must be less than end when step is greater than 0, or greater than end when step is less than 0.</td>
      <td>FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end</td>
      <td>Input</td>
      <td>End position of the range from which values are obtained.</td>
      <td>Ensure that start is less than end when step is greater than 0, or that start is greater than end when step is less than 0.</td>
      <td>FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>step</td>
      <td>Input</td>
      <td>Stride for obtaining values.</td>
      <td>Ensure that step is not equal to 0, that is, start is not equal to end.</td>
      <td>FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The specified output tensor.</td>
      <td>-</td>
      <td>FLOAT16, FLOAT, DOUBLE, BFLOAT16, INT32, INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input start, end, step, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of start, end, step, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>start, end, and step do not satisfy the computation logic of range. That is, start is greater than end when step is greater than 0, or start is less than end when step is less than 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnRange

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
        <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnRangeGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, which contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: Returned status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRange defaults to a deterministic implementation.

Warning: When the input data type is float, use float to calculate the output size of out due to precision errors inherent to the data type. If you use double to calculate the output, the double result may be smaller than the float result, in which case the Tiling side will perform verification and issue an alarm.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_range.h"

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

#define ACLNN_ERR_PARAM_NULLPTR 161001

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
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output as required by the API.
    std::vector<int64_t> outShape = {
        8,
    };
    void* outDeviceAddr = nullptr;
    aclScalar* start = nullptr;
    aclScalar* end = nullptr;
    aclScalar* step = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    float startValue = 1.31f;
    float endValue = 9.97f;
    float stepValue = 1.17f;

    // Create the start aclScalar.
    start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
    CHECK_RET(start != nullptr, LOG_PRINT("ACLNN_ERR_PARAM_NULLPTR: start is null\n"); return ACLNN_ERR_PARAM_NULLPTR);
    // Create the end aclScalar.
    end = aclCreateScalar(&endValue, aclDataType::ACL_FLOAT);
    CHECK_RET(end != nullptr, LOG_PRINT("ACLNN_ERR_PARAM_NULLPTR: end is null\n"); return ACLNN_ERR_PARAM_NULLPTR);
    // Create the step aclScalar.
    step = aclCreateScalar(&stepValue, aclDataType::ACL_FLOAT);
    CHECK_RET(step != nullptr, LOG_PRINT("ACLNN_ERR_PARAM_NULLPTR: step is null\n"); return ACLNN_ERR_PARAM_NULLPTR);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRange
    ret = aclnnRangeGetWorkspaceSize(start, end, step, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRangeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory using the workspaceSize computed by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnRange
    ret = aclnnRange(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRange failed. ERROR: %d\n", ret); return ret);

    // 4. Synchronously wait for task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result from device memory to host memory
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar
    aclDestroyScalar(start);
    aclDestroyScalar(end);
    aclDestroyScalar(step);
    aclDestroyTensor(out);

    // 7. Release device resources
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
