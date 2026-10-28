# aclnnMean

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T01:40:00.952Z -->

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

- Description: Computes the mean of a tensor along specified dimensions.
- Formula:

$$
\frac{1}{|\text{dim}|} \sum_{\text{dim}} X
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnMeanGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process. Then, call the "aclnnMean" API to perform the computation.

```Cpp
aclnnStatus aclnnMeanGetWorkspaceSize(
  const aclTensor*     self,
  const aclIntArray*   dim,
  bool                 keepDim,
  aclDataType          dtype,
  aclTensor*           out,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMean(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMeanGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">X in the formula.</td>
      <td class="tg-0pky">        <ul>
          <li>If the input is an empty tensor, the output type cannot be the complex data types COMPLEX64 or COMPLEX128.</li>
          <li>The data type must be convertible to the data type of dtype (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>)</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, COMPLEX128</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">dim (aclIntArray*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input array.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">keepDim (bool)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Whether the dimension of the reduce axis is retained.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">dtype (aclDataType)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the data type of the output Tensor.</td>
      <td class="tg-0pky">The data type must be convertible to the data type of out (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, COMPLEX128</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output tensor.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, COMPLEX64, COMPLEX128</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the Device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the computation process of the operator.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 285px">
  <col style="width: 123px">
  <col style="width: 740px">
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
      <td>self, out, and dim are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of self, out, and dim are out of the supported range, or dtype specifies an unsupported data type.</td>
    </tr>
    <tr>
      <td>The data formats of self and out are out of the supported range.</td>
    </tr>
    <tr>
      <td>The dimensions in the dim array exceed the dimension range of the input tensor.</td>
    </tr>
    <tr>
      <td>Duplicate elements exist in the dim array.</td>
    </tr>
    <tr>
      <td>The out shape does not match the actual shape.</td>
    </tr>
    <tr>
      <td>The data type of self cannot be converted to the data type of dtype.</td>
    </tr>
    <tr>
      <td>The data type of dtype cannot be converted to the data type of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnMean

- **Parameters:**

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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnMeanGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMean defaults to a deterministic implementation.
<!-- npu="950" id8 -->
- Batch consistency description:
  - <term>Ascend 950PR&950DT products</term>: defaults to a non-Batch-consistent implementation. Batch consistency can be enabled by calling aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3). Once enabled, the computation results of non-reduction axes are independent of the batch size or position; reduction axes do not support Batch consistency. Enabling Batch consistency may degrade performance.
<!-- end id8 -->

## Example

The sample code is as follows and is for reference only. For details about how to compile and run it, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mean.h"

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
  // Call aclrtMalloc to allocate memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);

  // Call the aclCreateIntArray API to create an aclIntArray
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}

int PrepareInputAndOutput(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& outShape, void** selfDeviceAddr, aclTensor** self, aclIntArray** dim,
    void** outDeviceAddr, aclTensor** out)
{
    std::vector<int64_t> selfHostData = {2, 3, 5, 8, 4, 12, 6, 7};
    std::vector<int64_t> outHostData = {2, 3};
    std::vector<int64_t> dimData = {1, 2};

    // Create the self aclTensor
    auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_INT64, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the dim aclIntArray
    ret = CreateAclIntArray(dimData, dim);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor
    ret = CreateAclTensor(outHostData, outShape, outDeviceAddr, aclDataType::ACL_INT64, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

void ReleaseTensorAndIntArray(aclTensor* self, aclIntArray* dim, aclTensor* out)
{
    aclDestroyTensor(self);
    aclDestroyIntArray(dim);
    aclDestroyTensor(out);
}

void ReleaseDevice(
    void* selfDeviceAddr, void* outDeviceAddr, uint64_t workspaceSize, void* workspaceAddr, aclrtStream stream,
    int32_t deviceId)
{
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API definition.
  std::vector<int64_t> selfShape = {2, 2, 2};
  std::vector<int64_t> outShape = {2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* dim = nullptr;
  aclTensor* out = nullptr;

  ret = PrepareInputAndOutput(selfShape, outShape, &selfDeviceAddr, &self, &dim, &outDeviceAddr, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  bool keepdim = false;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMean.
  ret = aclnnMeanGetWorkspaceSize(self, dim, keepdim, aclDataType::ACL_INT64, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMean.
  ret = aclnnMean(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMean failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the specific API definition.
  ReleaseTensorAndIntArray(self, dim, out);

  // 7.Release the device resources. Modify it based on the specific API definition.
  ReleaseDevice(selfDeviceAddr, outDeviceAddr, workspaceSize, workspaceAddr, stream, deviceId);

  return 0;
}
```
