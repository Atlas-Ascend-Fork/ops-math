# aclnnStridedSliceGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:40:28.478Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: The backward operator of StridedSlice, which maps the slice gradient tensor dy back to the corresponding positions of the original tensor shape and fills the positions not covered by the slice with zeros.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnStridedSliceGradGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnStridedSliceGrad" API to execute the computation.

```Cpp
aclnnStatus aclnnStridedSliceGradGetWorkspaceSize(
    const aclIntArray  *shape,
    const aclIntArray  *begin,
    const aclIntArray  *end,
    const aclIntArray  *strides,
    const aclTensor    *dy,
    int64_t             beginMask,
    int64_t             endMask,
    int64_t             ellipsisMask,
    int64_t             newAxisMask,
    int64_t             shrinkAxisMask,
    const aclTensor    *out,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnStridedSliceGrad(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnStridedSliceGradGetWorkspaceSize

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>shape (aclIntArray*)</td>
      <td>Input</td>
      <td>Shape of the original input tensor, that is, the shape of output out.</td>
      <td><ul><li>The length and begin/end/strides are the same.</li><li>Each element value is ≥ 0 (0 indicates an empty tensor).</li><li>The rank does not exceed 8 after removing dimensions of length 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>begin (aclIntArray*)</td>
      <td>Input</td>
      <td>Slice start index, corresponding to the begin parameter of the corresponding forward StridedSlice.</td>
      <td><ul><li>The length and shape are the same.</li><li>Negative indices are supported (counted from the end).</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end (aclIntArray*)</td>
      <td>Input</td>
      <td>End index of the slice (excluded), corresponding to the end parameter of the corresponding forward StridedSlice.</td>
      <td><ul><li>The length and shape are the same.</li><li>Negative indices are supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides (aclIntArray*)</td>
      <td>Input</td>
      <td>Slicing strides, which correspond to the strides parameter of the corresponding forward StridedSlice.</td>
      <td><ul><li>The length and shape length are the same.</li><li>Element values cannot be 0, and negative values (reverse slicing) are supported.</li><li>The strides of dimensions set in shrinkAxisMask must be 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Output gradient tensor of the slice operation.</td>
      <td><ul><li>dy.shape must be equal to the output shape after the forward StridedSlice operation is performed on the original shape.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beginMask (int64_t)</td>
      <td>Input</td>
      <td>Bitmask. When the i-th bit is 1, begin of the dimension is ignored and the slice starts from the first element of that dimension.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>endMask (int64_t)</td>
      <td>Input</td>
      <td>Bitmask. When the i-th bit is 1, end of the dimension is ignored and the slice extends to the end of that dimension.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsisMask (int64_t)</td>
      <td>Input</td>
      <td>Bitmask that marks the ellipsis dimension. Marked dimensions are fully retained (equivalent to ":").</td>
      <td><ul><li>At most 1 bit is set to 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>newAxisMask (int64_t)</td>
      <td>Input</td>
      <td>Bitmask. If the i-th bit is 1, a new dimension (length 1) is inserted at the corresponding position.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrinkAxisMask (int64_t)</td>
      <td>Input</td>
      <td>Bitmask. When bit i is 1, the corresponding dimension is squeezed (takes a single element) in the output. The strides of the set dimensions must be 1.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Backward propagation result tensor, with the same shape as the input of the original forward operator (that is, the shape specified by the shape parameter).</td>
      <td><ul><li>The data type is the same as that of dy.</li><li>Positions not covered by the slice are filled with 0.</li></ul></td>
      <td>Same as dy</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the Device-side-applied workspace.</td>
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

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
      <td>dy or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of dy or out is not supported.</td>
    </tr>
  </tbody></table>

## aclnnStridedSliceGrad

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
      <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size side-applied on the Device, obtained by the first-phase API aclnnStridedSliceGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody></table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnStridedSliceGrad defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice_grad.h"

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
    // Boilerplate for AscendCL initialization.
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy host data to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. Refer to the AscendCL public API list.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs.
    //
    // Use case description:
    //   The forward StridedSlice slices [0:2, 1:3, 2:4] from a tensor with shape=[3,4,5] (all strides are 1),
    //   the forward output shape = [2, 2, 2].
    //   This backward operator scatters the gradients in dy (shape=[2,2,2]) back to output (shape=[3,4,5]),
    //   filling the positions not covered by the slice with 0.
    //
    // Parameters:
    //   shape   = [3, 4, 5]   — shape of the original forward input, i.e., the shape of output
    //   begin   = [0, 1, 2]
    //   end     = [2, 3, 4]
    //   strides = [1, 1, 1]
    //   mask is all 0

    std::vector<int64_t> shapeData = {3, 4, 5};
    std::vector<int64_t> beginData = {0, 1, 2};
    std::vector<int64_t> endData = {2, 3, 4};
    std::vector<int64_t> stridesData = {1, 1, 1};

    std::vector<int64_t> dyShape = {2, 2, 2};
    std::vector<int64_t> outputShape = {3, 4, 5};

    // dy gradient data (float, all 1.0)
    std::vector<float> dyHostData(GetShapeSize(dyShape), 1.0f);
    // output is initialized to 0
    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0f);

    void* dyDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* output = nullptr;

    // Create dy aclTensor.
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create value-dependent inputs (aclIntArray).
    aclIntArray* shape = aclCreateIntArray(shapeData.data(), shapeData.size());
    aclIntArray* begin = aclCreateIntArray(beginData.data(), beginData.size());
    aclIntArray* end = aclCreateIntArray(endData.data(), endData.size());
    aclIntArray* strides = aclCreateIntArray(stridesData.data(), stridesData.size());
    CHECK_RET(shape != nullptr && begin != nullptr && end != nullptr && strides != nullptr,
              LOG_PRINT("aclCreateIntArray failed.\n");
              return -1);

    // mask parameter (all 0s indicate that no special slicing semantics is enabled)
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    // 3. Call the CANN operator library API (two-phase API)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // Call the aclnnStridedSliceGrad first-phase API
    ret = aclnnStridedSliceGradGetWorkspaceSize(shape, begin, end, strides, dy, beginMask, endMask, ellipsisMask,
                                                newAxisMask, shrinkAxisMask, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the aclnnStridedSliceGrad second-phase API
    ret = aclnnStridedSliceGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Copy the result from the device side back to the host side and print it
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclIntArray and aclTensor
    aclDestroyIntArray(shape);
    aclDestroyIntArray(begin);
    aclDestroyIntArray(end);
    aclDestroyIntArray(strides);
    aclDestroyTensor(dy);
    aclDestroyTensor(output);

    // 7. Release device resources
    aclrtFree(dyDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
