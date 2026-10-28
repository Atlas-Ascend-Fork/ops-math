# aclnnSort

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:16:33.423Z -->

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

- Description: Sorts the elements in the input tensor in ascending or descending order along the specified dimension, and returns the sorted values (valuesOut) and the corresponding indices (indicesOut).

- Formula:

$$
valuesOut[\mathbf{i}_{\neg dim}, j] = \text{sort}_{dim}(self[\mathbf{i}_{\neg dim}, :])[j]
$$

$$
indicesOut[\mathbf{i}_{\neg dim}, j] = \text{argsort}_{dim}(self[\mathbf{i}_{\neg dim}, :])[j]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSortGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process, and then call the "aclnnSort" API to perform the computation.

```Cpp
aclnnStatus aclnnSortGetWorkspaceSize(
  const aclTensor *self,
  bool             stable,
  int64_t          dim,
  bool             descending,
  aclTensor       *valuesOut,
  aclTensor       *indicesOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSort(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnSortGetWorkspaceSize

- **Parameters**:

  <table style="table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 167px">
  <col style="width: 127px">
  <col style="width: 298px">
  <col style="width: 217px">
  <col style="width: 326px">
  <col style="width: 126px">
  <col style="width: 140px">
  <col style="width: 147px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stable</td>
      <td>Input</td>
      <td>Specifies whether the sorting is stable. True indicates stable sorting, and False indicates unstable sorting.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Dimension used as the sorting criterion.</td>
      <td>Range: [-self.dim(), self.dim()-1], where self.dim() is the dimension of the input tensor.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>descending</td>
      <td>Input</td>
      <td>Controls the sort order. True indicates descending order, and False indicates ascending order.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valuesOut</td>
      <td>Output</td>
      <td>Sorting result of the tensor along the specified dimension.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, UINT16, UINT32, UINT64</td>
      <td>ND</td>
      <td>Same as self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indicesOut</td>
      <td>Output</td>
      <td>Represents the indices of each element in the original tensor after sorting.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>Same as self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The data types UINT16, UINT32, and UINT64 are not supported for self.
    - When the data type of self is BFLOAT16, the axis specified by the dim parameter cannot be 1.
    - The data type of valuesOut does not support UINT16, UINT32, or UINT64.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: valuesOut does not support the DOUBLE data type.
  <!-- end id8 -->

- **Return value**:

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
      <td>The input self, valuesOut, or indicesOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of self, valuesOut, or indicesOut is not within the supported range, or the shapes do not match.</td>
    </tr>
    <tr>
      <td>The value of dim is out of the dimension range of input tensor self.</td>
    </tr>
  </tbody>
  </table>

## aclnnSort

- **Parameters**:

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnSortGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSort defaults to a deterministic implementation.

- When the data type of self is not FLOAT, FLOAT16, or BFLOAT16, an excessively large tensor size may cause operator execution to time out (an aicpu error is reported, with reason=[aicpu timeout]). The maximum size limits for specific types (strongly related to the remaining memory of the machine) are as follows:
  - INT64: 150000000
  - UINT8, INT8, INT16, INT32 types: 725000000

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sort.h"

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
  // Boilerplate. Initialize resources.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. See the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  bool stable = false;
  int64_t dim = 0;
  bool descending = false;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> outValuesShape = {3, 4};
  std::vector<int64_t> outIndicesShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* outValuesDeviceAddr = nullptr;
  void* outIndicesDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* outValues = nullptr;
  aclTensor* outIndices = nullptr;
  std::vector<int64_t> selfHostData = {7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> outValuesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> outIndicesHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the outValues and outIndices aclTensor.
  ret = CreateAclTensor(outValuesHostData, outValuesShape, &outValuesDeviceAddr, aclDataType::ACL_INT64, &outValues);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outIndicesHostData, outIndicesShape, &outIndicesDeviceAddr, aclDataType::ACL_INT64, &outIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSort.
  ret = aclnnSortGetWorkspaceSize(self, stable, dim, descending, outValues, outIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSortGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSort.
  ret = aclnnSort(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSort failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outValuesShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outValuesDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result values [%ld] is: %ld\n", i, resultData[i]);
  }

  auto size2 = GetShapeSize(outIndicesShape);
  std::vector<int64_t> resultData2(size2, 0);
  ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), outIndicesDeviceAddr,
                    size * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size2; i++) {
    LOG_PRINT("result indices [%ld] is: %ld\n", i, resultData2[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(outValues);
  aclDestroyTensor(outIndices);

  // 7. Release the device resources. Modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outValuesDeviceAddr);
  aclrtFree(outIndicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
