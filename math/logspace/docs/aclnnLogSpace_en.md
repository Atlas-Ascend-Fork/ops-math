# aclnnLogSpace

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T00:38:48.289Z -->

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

- Description: Creates a 1D tensor of size $\text{steps}$, whose values are evenly spaced on a logarithmic scale from $\text{base}^\text{start}$ to $\text{base}^\text{end}$, inclusive of endpoints, with $\text{base}$ as the base.

- Formula:

$$
\text{result} = \left(\text{base}^\text{start},\ \text{base}^{\left(\text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)},\ \ldots,\ \text{base}^{\left(\text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1}\right)},\ \text{base}^\text{end}\right)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnLogSpaceGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnLogSpace" to perform the computation.

```cpp
aclnnStatus aclnnLogSpaceGetWorkspaceSize(
  const aclScalar*    start,
  const aclScalar*    end,
  int64_t             steps,
  double              base,
  const aclTensor*    result,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnLogSpace(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnLogSpaceGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 150px">
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimensions(shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>start (aclScalar*)</td>
      <td>Input</td>
      <td>First input of LogSpace, specifying the starting exponent of the log sequence.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, UINT64, UINT32, UINT16, UINT8, INT64, INT32, INT16, INT8, BOOL</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>end (aclScalar*)</td>
      <td>Input</td>
      <td>Second input of LogSpace, specifying the ending exponent of the log sequence.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE, UINT64, UINT32, UINT16, UINT8, INT64, INT32, INT16, INT8, BOOL</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>steps (int64_t)</td>
      <td>Input</td>
      <td>Number of elements in the sequence.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
   <tr>
      <td>base (double)</td>
      <td>Input</td>
      <td>Base of the logarithmic space.</td>
      <td>-</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>result (aclTensor*)</td>
      <td>Output</td>
      <td>Output of LogSpace, the logarithmically spaced sequence tensor.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
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
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed start, end, steps, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of out is not within the supported range.</td>
    </tr>
    <tr>
      <td>steps is less than 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnLogSpace

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
      <td>The workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the Device, obtained by the first-phase API aclnnLogSpaceGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation:**

  - aclnnLogSpace defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <math.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_logspace.h"

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
  // Boilerplate code for initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateOutputAclTensor(
    const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(),1);
    for (int64_t i = shape.size() - 2; i >=0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize them as required by the API.
  void* outDeviceAddr = nullptr;
  aclScalar* start = nullptr;
  aclScalar* end = nullptr;
  aclTensor* out = nullptr;

  float startValue = 0.0f;  //Start exponent
  float endValue = 2.0f;    //End exponent
  int64_t steps = 5;    //Number of steps
  float base = 10.0;    //Base

  std::vector<int64_t> shape = {steps};

  // Create start aclScalar.
  start = aclCreateScalar(&startValue, aclDataType::ACL_FLOAT);
  CHECK_RET(start != nullptr, LOG_PRINT("create start scalar failed\n"); return -1);
  // Create end aclScalar.
  end = aclCreateScalar(&endValue, aclDataType::ACL_FLOAT);
  CHECK_RET(end != nullptr, LOG_PRINT("create end scalar failed\n"); return -1);
  // Create out aclTensor.
  ret = CreateOutputAclTensor<float>(shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create output tensor failed\n"); return ret);
  // 3. Call a CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnLogSpace first-phase API
  ret = aclnnLogSpaceGetWorkspaceSize(start, end, steps, base, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSpaceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnLogSpace second-phase API
  ret = aclnnLogSpace(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSpace failed. ERROR: %d\n", ret); return ret);

  // 4. Synchronize and wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on device memory to the host
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar
  aclDestroyScalar(start);
  aclDestroyScalar(end);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify as required by the specific API definition
  aclrtFree(outDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
