# aclnnAddLora

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T23:10:19.468Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: not supported
<!-- end id6 -->

## Function

- Description:

  Multiplies the input x with the corresponding weightA and weightB based on the input indices, and then accumulates the result onto the input y and outputs it.

- Formula:

  Given an input tensor x whose last dimension has a length of 2d, the AddLora function performs the following computation:

  1. Rearrange x according to the indices in indices, so that x corresponding to the same group of weights are placed together.

  2. Iterate over each LoRA group, and perform matrix multiplication on the corresponding x and weightA respectively:

     $$
     Z1 = x_{i} \cdot weightA[i, layerIdx, :, :]
     $$
  3. The resulting `Z1` then undergoes matrix multiplication with weightB:

     $$
     Z2 = Z1 \cdot weightB[i, layerIdx, :, :] \times scale
     $$
  4. Finally, accumulate the output `Z2` onto y:

     $$
     \text{out} = y[:, yOffset: yOffset+ySliceSize] + Z2
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAddLoraGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the calculation process, and then call the "aclnnAddLora" API to perform the computation.

```Cpp
aclnnStatus aclnnAddLoraGetWorkspaceSize(
    const aclTensor *y,
    const aclTensor *x,
    const aclTensor *weightB,
    const aclTensor *indices,
    const aclTensor *weightAOptional,
    int64_t          layerIdx,
    double           scale,
    int64_t          yOffset,
    int64_t          ySliceSize,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLora(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAddLoraGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>y (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the tensor to be updated by accumulation, y in the formula.</td>
      <td><ul><li>2D shape: [B, H3]. H3 must be an integer multiple of 16, and the range of H3 must support 1-131072.</li><li>The first dimension needs to be consistent with the first dimension of x, both represented by `B`.</li><li>Does not support empty tensors.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input tensor before grouping, x in the formula.</td>
      <td><ul><li>2 dimensions in shape: [B, H1], and H1 is an integer multiple of 16.</li><li>Does not support empty tensors.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightB (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the second weight matrix for matrix multiplication, that is, weightB in the formula.</td>
      <td><ul><li>4 dimensions in shape: [W, L, H2, R]. The value of the third dimension needs to be less than or equal to the second dimension of y (H2 ≤ H3), and H2 is an integer multiple of 16; the range of H2 must support 1-131072. The range of R must support 1-128, and R is an integer multiple of 16.</li><li>Does not support empty tensors.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND, NZ</td>
      <td>4</td>
      <td>√</td>
    </tr>
     <tr>
      <td>indices (aclTensor*)</td>
      <td>Input</td>
      <td>Identifies the group index of input x, that is, the input indices in the formula.</td>
      <td><ul><li>1-D shape: [B].</li><li>The first dimension must keep consistent with the first dimensions of x and y, both denoted as `B`.</li><li>This parameter does not support empty tensors.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightAOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the first weight matrix for matrix multiplication. If it is empty, the first matrix multiplication is skipped. weightA in the formula.</td>
      <td><ul><li>shape with 4 dimensions: [W, L, R, H1]. The first two dimensions must keep consistent with the first two dimensions of `weightB`, represented by `W` and `L`, where the range of W is 1-32 and the range of L is 1-32. The third dimension must keep consistent with the fourth dimension of `weightB`, both represented by `R`. The fourth dimension must keep consistent with the second dimension of `x`, both represented by `H1`, and must be an integer multiple of 16.</li><li>Does not support empty tensors.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND, NZ</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>layerIdx (int64_t)</td>
      <td>Input</td>
      <td>Indicates the layer index, layerIdx in the formula.</td>
      <td>The value needs to be less than the second dimension L of weightB.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (double)</td>
      <td>Input</td>
      <td>Indicates the scaling coefficient, that is, scale in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOffset (int64_t)</td>
      <td>Input</td>
      <td>Indicates the offset for updating y, that is, yOffset in the formula.</td>
      <td>The value needs to be less than the second dimension H3 of y.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ySliceSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the range for updating y, namely ySliceSize in the formula.</td>
      <td>The value needs to be less than or equal to the second dimension H3 of y.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, the output out in the formula.</td>
      <td><ul><li>The data type of the output must keep consistent with that of the input.</li><li>The dimensions of the output shape must keep consistent with those of the input y shape.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
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

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: weightB and weightAOptional support the ND data format.
  <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed input parameter (x, y, weightB, indices) or output parameter out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input parameter (x, y, weightB, indices) or output parameter is not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The shape information of the input tensors does not match. (For details, see Parameters.)</td>
    </tr>
    <tr>
      <td>The shape information of the input tensor is currently not supported. (For details, see Parameters.)</td>
    </tr>
  </tbody>
  </table>

## aclnnAddLora

- **Parameters:**

  <table style="table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnAddLoraGetWorkspaceSize.</td>
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
  - aclnnAddLora defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_lora.h"

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
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, initialization.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs as required by the API.
  int32_t batchSize = 1;
  int32_t H1 = 16;
  int32_t H2 = 16;
  int32_t R = 16;
  int32_t loraNum = 1;
  int32_t layerNum = 1;

  std::vector<int64_t> xShape = {batchSize, H1};
  std::vector<int64_t> yShape = {batchSize, H2};
  std::vector<int64_t> weightBShape = {loraNum, layerNum, H2, R};
  std::vector<int64_t> indicesShape = {batchSize};
  std::vector<int64_t> weightAShape = {loraNum, layerNum, R, H1};
  std::vector<int64_t> outShape = {batchSize, H2};

  std::vector<float> xHostData(batchSize * H1, 1);
  std::vector<float> yHostData(batchSize * H2, 1);
  std::vector<float> weightBHostData(loraNum * layerNum * H2 * R, 1);
  std::vector<float> indicesHostData(batchSize, 0);
  std::vector<float> weightAHostData(loraNum * layerNum * R * H1, 1);
  std::vector<float> outHostData(batchSize * H2, 1);

  void* xInputDeviceAddr = nullptr;
  void* yInputDeviceAddr = nullptr;
  void* weightBInputDeviceAddr = nullptr;
  void* indicesInputDeviceAddr = nullptr;
  void* weightAInputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* xInput = nullptr;
  aclTensor* yInput = nullptr;
  aclTensor* weightBInput = nullptr;
  aclTensor* indicesInput = nullptr;
  aclTensor* weightAInput = nullptr;
  aclTensor* out = nullptr;

  // Create input x.
  ret = CreateAclTensor(xHostData, xShape, &xInputDeviceAddr, aclDataType::ACL_FLOAT16, &xInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input y.
  ret = CreateAclTensor(yHostData, yShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT16, &yInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input weightB.
  ret = CreateAclTensor(weightBHostData, weightBShape, &weightBInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightBInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input indices.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesInputDeviceAddr, aclDataType::ACL_INT32, &indicesInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input weightA.
  ret = CreateAclTensor(weightAHostData, weightAShape, &weightAInputDeviceAddr, aclDataType::ACL_FLOAT16, &weightAInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t layer_idx = 0;
  double scale = 1.0;
  int64_t y_offset = 0;
  int64_t y_slice_size = H2;

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnAddLora.
  ret = aclnnAddLoraGetWorkspaceSize(yInput, xInput, weightBInput, indicesInput, weightAInput, layer_idx, scale, y_offset, y_slice_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLoraGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnAddLora.
  ret = aclnnAddLora(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLora failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(xInput);
  aclDestroyTensor(yInput);
  aclDestroyTensor(weightBInput);
  aclDestroyTensor(indicesInput);
  aclDestroyTensor(weightAInput);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(xInputDeviceAddr);
  aclrtFree(yInputDeviceAddr);
  aclrtFree(weightBInputDeviceAddr);
  aclrtFree(indicesInputDeviceAddr);
  aclrtFree(weightAInputDeviceAddr);
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
