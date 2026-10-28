# aclnnRepeat

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:29:09.385Z -->

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

- Description: Replicates the input tensor along each dimension by the number of times specified in repeats.
- Example: Assume the input tensor is [[a,b],[c,d],[e,f]] with shape [3,2], and repeats is (2,4). The resulting tensor has shape [6,8] with the following values:

  ```text
  >>> x = torch.tensor([[a,b],[c,d],[e,f]])
  >>> x.repeat(2,4)
  tensor([[a,b,a,b,a,b,a,b],
          [c,d,c,d,c,d,c,d],
          [e,f,e,f,e,f,e,f],
          [a,b,a,b,a,b,a,b],
          [c,d,c,d,c,d,c,d],
          [e,f,e,f,e,f,e,f],
          ])
  ```

- When repeats is (2,4,2), meaning the number of elements in repeats is greater than the number of dimensions in the tensor, the output tensor is equivalent to the following operation: first, expand the shape of the input tensor to the same number of dimensions as repeats: [1,3,2], then expand according to the corresponding dimensions and the values of repeats, resulting in an output tensor shape of [2,12,4], as shown below:

  ```text
  >>> x.repeat(2,4,2)
  tensor([[[a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f]],

          [[a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f],
          [a,b,a,b],
          [c,d,c,d],
          [e,f,e,f]]])
  ```

  - The following conditions must be met during computation:
    - The number of parameters in repeats must not be less than the number of dimensions of the input tensor.
    - The values in repeats must be greater than or equal to 0.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context//two_phase_api.md). You must first call the "aclnnRepeatGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnRepeat" API to perform the computation.

```cpp
aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor   *self,
    const aclIntArray *repeats,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnRepeat(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRepeatGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 290px">
  <col style="width: 110px">
  <col style="width: 150px">
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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT, DOUBLE, FLOAT16, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64, BOOL, BFLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>≤8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats (aclIntArray*)</td>
      <td>Input</td>
      <td>-</td>
      <td>Number of times to repeat the input Tensor along each dimension. The number of repeats must not exceed 8.<br>Currently, the use case of repeating along more than 4 dimensions at the same time is not supported. For details, see Constraints.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>Same as self.</td>
      <td>ND</td>
      <td>≤8</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data types HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN are not supported.
  <!-- end id7 -->
  <!-- npu="910,310p" id8 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data types BFLOAT16, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN are not supported.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
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
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type and data format of self and out are out of the supported range.</td>
    </tr>
    <tr>
      <td>The types of self and out do not match.</td>
    </tr>
    <tr>
      <td>The number of elements in repeats is less than the dimensions of the input tensor.</td>
    </tr>
    <tr>
      <td>repeats contains values less than 0.</td>
    </tr>
    <tr>
      <td>The number of dimensions of self exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of elements in repeats exceeds 8.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>The kernel fails to execute because the intermediate result is null.</td>
    </tr>
    <tr>
      <td>The repeat operation is performed on more than 4 dimensions at a time.</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeat

- **Parameters**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device. The value is obtained by the first-phase API aclnnRepeatGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that specifies the task to execute.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRepeat defaults to a deterministic implementation.

The broadcast kernel used internally by repeat supports a maximum of 8 dimensions. Cases where the dimension exceeds 8 after broadcasting are not supported currently. Details:
  Limitation 1. When repeat is required on the first axis, up to 4 dimensions can be repeated at the same time (that is, the number of non-1 values in repeats does not exceed 4).

  ```text
   x.repeat(2, 3, 4, 5, 6)  # Not supported. A check error is reported because the first axis is repeated 2 times and there are 5 non-1 repeat parameters in total.
   x.repeat(2, 3, 1, 5, 6)  # Supported: the first axis is repeated 2 times, with 4 non-1 repeat parameters
  ```

  Limitation 2. When the first axis does not need to be repeated, repeat operations on a maximum of 3 dimensions are supported at the same time (that is, the number of non-1 parameters in repeats does not exceed 3).

  ```text
   x.repeat(1, 3, 4, 5, 6)  # Not supported: a check error is reported; the first axis is repeated 1 time, with 4 non-1 repeat parameters
   x.repeat(1, 3, 1, 5, 6)  # Supported: the first axis is repeated 1 time, with 3 non-1 repeat parameters
  ```

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
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
  // Call aclrtMalloc to apply for memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray) {
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the ACL API documentation.
  // Set deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> selfShape = {3, 2};
  std::vector<int64_t> outShape = {2, 12, 4};
  std::vector<int64_t> repeatsArray = {2, 4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* repeat = nullptr;
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 1);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the repeat aclIntArray.
  aclCreateIntArrayP(repeatsArray, &repeat);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeat.
  ret = aclnnRepeatGetWorkspaceSize(self, repeat, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeat.
  ret = aclnnRepeat(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeat failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyIntArray(repeat);

  // 7. Release device resources. Modify it according to the specific API definition.
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
