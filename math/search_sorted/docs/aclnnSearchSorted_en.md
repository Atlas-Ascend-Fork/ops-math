# aclnnSearchSorted

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:00:21.971Z -->

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

- Description: Finds the positions where given tensor values (self) should be inserted in a sorted tensor (sortedSequence). Returns a tensor of the same size as self, where each element indicates the insertion position of the given value in the original tensor. If self is of Scalar type, see [aclnnSearchSorteds](./aclnnSearchSorteds_en.md).
- Formula: Let N be the length of the innermost sequence to be searched. For each input element x = self<sub>i</sub>:
  - When `right=false`, the left insertion point is returned:

  $$
  out_i=\min\{j\in[0,N]\mid sortedSequence_j\ge x\}
  $$

  - When `right=true`, the right insertion point is returned:

  $$
  out_i=\min\{j\in[0,N]\mid sortedSequence_j>x\}
  $$

  If no such $j$ exists, returns $N$.

## Function Prototype

Each operator uses a [two-phase APIs](../../../docs/en/context/two_phase_api.md). Call `aclnnSearchSortedGetWorkspaceSize` to obtain the workspace size required for computation and the executor containing the operator computation process. Then call `aclnnSearchSorted` to perform the computation.

```Cpp
aclnnStatus aclnnSearchSortedGetWorkspaceSize(
  const aclTensor* sortedSequence,
  const aclTensor* self,
  bool             outInt32,
  bool             right,
  const aclTensor* sorter,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSearchSorted(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnSearchSortedGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sortedSequence (aclTensor*)</td>
      <td>Input</td>
      <td>Sorted input tensor.</td>
      <td><ul><li>The data type must satisfy the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a> with <code>self</code>.</li><li>In the formula, <code>N</code> is the length of the innermost dimension of <code>sortedSequence</code>, and <code>sortedSequence<sub>j</sub></code> is the <code>j</code>-th element of the innermost dimension.</li></ul></td>
      <td>DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>No more than 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor for which insertion positions are to be found.</td>
      <td><ul><li>The data type of <code>sortedSequence</code> must satisfy the <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a> with that of this parameter.</li><li>In the formula, <code>x=self<sub>i</sub></code>, where <code>i</code> is the element index in <code>self</code>.</li></ul></td>
      <td>DOUBLE, FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>At most 8 dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outInt32(bool)</td>
      <td>Input</td>
      <td>Specifies whether to output INT32 results.</td>
      <td>Specifies the data type of output indexes.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right (bool)</td>
      <td>Input</td>
      <td>Specifies whether to return the left or right insertion point when equal values are hit.</td>
      <td><ul><li><code>false</code> corresponds to <code>sortedSequence<sub>j</sub> &ge; x</code> in the formula.</li><li><code>true</code> corresponds to <code>sortedSequence<sub>j</sub> &gt; x</code>.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sorter (aclTensor*)</td>
      <td>Input</td>
      <td>Specifies the order of <code>sortedSequence</code> elements.</td>
      <td><ul><li>The shape must be the same as <code>sortedSequence</code>.</li><li>The value range of elements is [0, innermost dim-1].</li><li>If <code>sorter</code> is passed, <code>sortedSequence<sub>j</sub></code> in the formula takes values in the order specified by <code>sorter</code>.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>Same as sortedSequence</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output result of insertion positions.</td>
      <td><ul><li>The shape must be the same as that of <code>self</code>.</li><li><code>out<sub>i</sub></code> in the formula corresponds to the <code>i</code>th element of the output <code>out</code>.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>Same as self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
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
      <td>A null pointer exists in sortedSequence, self, or out.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of sortedSequence and self are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data type of out conflicts with the meaning of the outInt32 value.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed when sortedSequence and self have different data types.</td>
    </tr>
    <tr>
      <td>The input sorter is not of the INT64 type.</td>
    </tr>
    <tr>
      <td>The shape of out is different from that of self, and the shape of sorter is different from that of sortedSequence.</td>
    </tr>
    <tr>
      <td>When sortedSequence has more than one dimension and dimensions of self other than the last one are not equal to the corresponding dimensions of sortedSequence.</td>
    </tr>
  </tbody></table>

## aclnnSearchSorted

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size obtained by the first-phase API <code>aclnnSearchSortedGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, including the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: `aclnnSearchSorted` defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_searchsorted.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message,...)     \
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
  // Boilerplate. Resource initialization.
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

  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check based on your needs.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> sortedSequenceShape = {4, 4};
  std::vector<int64_t> sorterShape = {4, 4};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* sortedSequenceDeviceAddr = nullptr;
  void* sorterDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* sortedSequence = nullptr;
  aclTensor* sorter = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {1,3,6,8,10,11,14,16};
  std::vector<float> sortedSequenceHostData = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  std::vector<int64_t> sorterHostData = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sortedSequence aclTensor.
  ret = CreateAclTensor(sortedSequenceHostData, sortedSequenceShape, &sortedSequenceDeviceAddr, aclDataType::ACL_FLOAT, &sortedSequence);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sorter aclTensor.
  ret = CreateAclTensor(sorterHostData, sorterShape, &sorterDeviceAddr, aclDataType::ACL_INT64, &sorter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  bool outInt32 = false;
  bool right = false;
  // 3. Call the CANN operator library API. Change it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSearchSorted.
  ret = aclnnSearchSortedGetWorkspaceSize(sortedSequence, self, outInt32, right, sorter, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSortedGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnSearchSorted.
  ret = aclnnSearchSorted(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSearchSorted failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(sortedSequence);
  aclDestroyTensor(sorter);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(sortedSequenceDeviceAddr);
  aclrtFree(sorterDeviceAddr);
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
