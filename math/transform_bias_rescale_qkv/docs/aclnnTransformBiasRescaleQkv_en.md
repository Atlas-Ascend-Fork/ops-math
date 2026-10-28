# aclnnTransformBiasRescaleQkv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:30:34.430Z -->

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

- Description:
  The TransformBiasRescaleQkv operator is an API for processing query, key, and value vectors in the multi-head attention mechanism. It adjusts the bias and rescaling factor of these vectors to optimize the attention computation process.

- Formula:
  The element-wise computation process is shown in the formula:

  $$

  q_o=(q_i+q_{bias})/\sqrt{dim\_per\_head}\\

  $$

  $$

  k_o=k_i+k_{bias}\\

  $$

  $$

    v_o=v_i+v_{bias}

  $$

  Where:

  - dim_per_head is the dimension of each attention head.
  - q<sub>o</sub>, k<sub>o</sub>, and v<sub>o</sub> are the output elements of the query, key, and value vectors, respectively.
  - q<sub>i</sub>, k<sub>i</sub>, and v<sub>i</sub> are the input elements of the query, key, and value vectors, respectively.
  - q<sub>bias</sub>, k<sub>bias</sub>, and v<sub>bias</sub> are the input element biases of the query, key, and value vectors, respectively.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnTransformBiasRescaleQkvGetWorkspaceSize" API to obtain the input parameters and calculate the workspace size required by the computation flow, and then call the "aclnnTransformBiasRescaleQkv" API to perform the computation.

```Cpp
aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qkvBias,
    int64_t          numHeads,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnTransformBiasRescaleQkv(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnTransformBiasRescaleQkvGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1370px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>qkv (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, which is q<sub>i</sub>, k<sub>i</sub>, and v<sub>i</sub> in the formula.</td>
      <td>3D tensor with shape {B,T,3 * num_heads * dim_per_head}. B is the batch size, T is the sequence length, num_heads is the number of attention heads, and dim_per_head is the dimension of each attention head.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvBias(aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, which is q<sub>bias</sub>, k<sub>bias</sub>, and v<sub>bias</sub> in the formula.</td>
      <td><ul><li>Shape: {3 * num_heads * dim_per_head} 1D tensor.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeads (int64_t)</td>
      <td>Input</td>
      <td>Number of input heads.</td>
      <td>The value must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>qOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, q<sub>o</sub> in the formula.</td>
      <td>A 4D tensor with shape {B,num_heads,T,dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, k<sub>o</sub> in the formula.</td>
      <td>4D tensor with shape {B,num_heads,T,dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, v<sub>o</sub> in the formula.</td>
      <td>4D tensor with shape {B,num_heads,T,dim_per_head}.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 295px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>The input or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td><ul><li>The data type or data format of qkv or qkvBias is not supported.</li><li>The data types of qkv and qkvBias are different.</li><li>The shapes of qkv and qkvBias do not meet the requirements described in Parameters.</li></ul></td>
    </tr>
  </tbody></table>

## aclnnTransformBiasRescaleQkv

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device. The value is obtained by the first-phase API aclnnTransformBiasRescaleQkvGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnTransformBiasRescaleQkv defaults to a deterministic implementation.

- The data types of the input qkv and qkvBias must be consistent with those of the output qOut, kOut, and vOut.
- If the input is NaN, the output is also NaN. If the input is Inf, the output is also Inf.
- If the input is -Inf, the output is also -Inf.

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_transform_bias_rescale_qkv.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outPut result[%ld] is: %f\n", i, resultData[i]);
  }
}

void PrintInResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("input[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize.
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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
// 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
// Fill in deviceId based on the actual device.
int32_t deviceId = 0;
aclrtStream stream;
auto ret = Init(deviceId, &stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

// 2. Construct the input and output as required by the API.
// qkv
int64_t B = 3;
int64_t T = 4;
int64_t n = 3;
int64_t d = 16;
std::vector<int64_t> qkvShape = {B, T, 3 * n * d};
int qkvCount = B * T * 3 * n * d;
std::vector<float> qkvHostData(qkvCount, 1);

for (int i = 0; i < qkvCount; ++i) {
    qkvHostData[i] = i * 1.0;
}

void* qkvDeviceAddr = nullptr;
aclTensor* qkv = nullptr;
// Create the input aclTensor.
ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT, &qkv);
CHECK_RET(ret == ACL_SUCCESS, return ret);

// qkvBias
std::vector<int64_t> qkvBiasShape = {3 * n * d};
std::vector<float> qkvBiasHostData(3 * n * d, 0.5);

void* qkvBiasDeviceAddr = nullptr;
aclTensor* qkvBias = nullptr;
// Create the input aclTensor.
ret = CreateAclTensor(qkvBiasHostData, qkvBiasShape, &qkvBiasDeviceAddr, aclDataType::ACL_FLOAT, &qkvBias);
CHECK_RET(ret == ACL_SUCCESS, return ret);

std::vector<int64_t> outShape = {B, n, T, d};
std::vector<float> outHostData(qkvCount / 3, 1);
aclTensor* outQ = nullptr;
aclTensor* outK = nullptr;
aclTensor* outV = nullptr;
void* outQDeviceAddr = nullptr;
void* outKDeviceAddr = nullptr;
void* outVDeviceAddr = nullptr;

// Create the out aclTensor.
ret = CreateAclTensor(outHostData, outShape, &outQDeviceAddr, aclDataType::ACL_FLOAT, &outQ);
ret = CreateAclTensor(outHostData, outShape, &outKDeviceAddr, aclDataType::ACL_FLOAT, &outK);
ret = CreateAclTensor(outHostData, outShape, &outVDeviceAddr, aclDataType::ACL_FLOAT, &outV);

CHECK_RET(ret == ACL_SUCCESS, return ret);

// 3. Call the CANN operator library API. Replace the API name with the actual one.
uint64_t workspaceSize = 16 * 1024 * 1024;
aclOpExecutor* executor;

// LOG_PRINT("qkv input=====");
// PrintInResult(qkvShape, &qkvDeviceAddr);

// LOG_PRINT("qkvBias input=====");
// PrintInResult(qkvBiasShape, &qkvBiasDeviceAddr);

// Call the first-phase API of aclnnTransformBiasRescaleQkv.
ret = aclnnTransformBiasRescaleQkvGetWorkspaceSize(
qkv,
qkvBias,
n,
outQ,
outK,
outV,
&workspaceSize,
&executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkvGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// Allocate device memory based on the workspaceSize calculated by the first-phase API.
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}

// Call the second-phase API of aclnnTransformBiasRescaleQkv.
ret = aclnnTransformBiasRescaleQkv(
workspaceAddr,
workspaceSize,
executor,
stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransformBiasRescaleQkv failed. ERROR: %d\n", ret); return ret);

// 4. (boilerplate) Synchronize and wait for the task execution to complete.
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5.Obtain the values of the output, copy the result from device memory to host memory, and modify it based on the specific API definition.
LOG_PRINT("q output=====");
PrintOutResult(outShape, &outQDeviceAddr);

LOG_PRINT("k output=====");
PrintOutResult(outShape, &outKDeviceAddr);


LOG_PRINT("v output=====");
PrintOutResult(outShape, &outVDeviceAddr);

// 6.Release aclTensor and modify it based on the specific API definition.
aclDestroyTensor(qkv);
aclDestroyTensor(qkvBias);
aclDestroyTensor(outQ);
aclDestroyTensor(outK);
aclDestroyTensor(outV);

// 7.Release device resources and modify it based on the specific API definition.
aclrtFree(qkvDeviceAddr);
aclrtFree(qkvBiasDeviceAddr);

aclrtFree(outQDeviceAddr);
aclrtFree(outKDeviceAddr);
aclrtFree(outVDeviceAddr);

if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();

return 0;
}
```
