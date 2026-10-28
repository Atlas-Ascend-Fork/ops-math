# aclnnCummin

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:45:05.254Z -->

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

- Description: Computes the cumulative minimum of self, and returns the values along with their corresponding indices.

- Formula:
  valuesOut:

  $$
  valuesOut_{i} = min(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

  indicesOut:

  $$
  indicesOut_{i} = argmin(self_{1}, self_{2}, self_{3}, ...... , self_{i})
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnCumminGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnCummin" to execute the computation.

```cpp
aclnnStatus aclnnCumminGetWorkspaceSize(
  const aclTensor*    self,
  int64_t             dim,
  aclTensor*          valuesOut,
  aclTensor*          indicesOut,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```cpp
aclnnStatus aclnnCummin(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  aclrtStream       stream)
```

## aclnnCumminGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 220px">
    <col style="width: 120px">
    <col style="width: 350px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 100px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>The data type must be convertible to the data type of valuesOut.</td>
        <td>FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64, FLOAT16, BFLOAT16, BOOL</td>
        <td>ND</td>
        <td>No more than 8 dimensions.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Dimension along which the operation is performed.</td>
        <td>The value range is [-self.dim(), self.dim()-1).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>valuesOut (aclTensor*)</td>
        <td>Output</td>
        <td>Output tensor of the cumulative minimum values.</td>
        <td>The data type and shape must be the same as those of self.</td>
        <td>FLOAT,DOUBLE,UINT8,INT8,INT16,INT32,INT64,FLOAT16,BFLOAT16,BOOL</td>
        <td>ND</td>
        <td>Up to 8 dimensions.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut (aclTensor*)</td>
        <td>Output</td>
        <td>Output tensor for the indices corresponding to the cumulative minimum values.</td>
        <td>Data types INT32 and INT64 are supported. The shape must be the same as that of self.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>Up to 8 dimensions.</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
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

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, and <term>Atlas training products</term>: BFLOAT16 is not supported.
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
      <td>self, valuesOut, or indicesOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of self, valuesOut, and indicesOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of self, valuesOut, and indicesOut are inconsistent.</td>
    </tr>
    <tr>
      <td>When self.dim()=0, the value range of dim is not within [-1, 0]; when self.dim()>0, the value range of dim is not within [-self.dim(), self.dim()-1].</td>
    </tr>
    <tr>
      <td>The data type of self cannot be converted to the data type of valuesOut.</td>
    </tr>
    <tr>
      <td>The dimensions of self, valuesOut, and indicesOut are greater than 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnCummin

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
        <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnCumminGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, which contains the computation flow of the operator.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream where the task is executed.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnCummin defaults to a deterministic implementation.

- When the input self data type is int32, due to instruction feature constraints, precision is guaranteed only when values are within [-16777215, 16777215].

## Example

The following sample code is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cummin.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Construct them based on the API.
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> valuesOutShape = {2, 4};
  std::vector<int64_t> indicesOutShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {3.0, 3.0, 2.0, 1.0, 4.0, 2.0, 6.0, 7.0};
  std::vector<float> valuesOutHostData(8, 0.0);
  std::vector<int64_t> indicesOutHostData(8, 0);
  int64_t dim = 0;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnCummin.
  ret = aclnnCumminGetWorkspaceSize(self, dim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCumminGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnCummin.
  ret = aclnnCummin(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCummin failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device to the host. Modify the code based on the specific API definition.
  auto valuesSize = GetShapeSize(valuesOutShape);
  std::vector<float> valuesResultData(valuesSize, 0);
  ret = aclrtMemcpy(valuesResultData.data(), valuesResultData.size() * sizeof(valuesResultData[0]), valuesOutDeviceAddr,
                    valuesSize * sizeof(valuesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy values result from device to host failed. ERROR: %d\n", ret); return ret);

  auto indicesSize = GetShapeSize(indicesOutShape);
  std::vector<int64_t> indicesResultData(indicesSize, 0);
  ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesOutDeviceAddr,
                    indicesSize * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("Values result:\n");
  for (int64_t i = 0; i < valuesSize; i++) {
    LOG_PRINT("valuesResult[%ld] is: %f\n", i, valuesResultData[i]);
  }

  LOG_PRINT("Indices result:\n");
  for (int64_t i = 0; i < indicesSize; i++) {
    LOG_PRINT("indicesResult[%ld] is: %ld\n", i, indicesResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
