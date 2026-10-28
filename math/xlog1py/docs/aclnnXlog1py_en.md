# aclnnXlog1py

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T02:31:41.299Z -->

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

- Description: Computes x * log(1 + y). The result is 0 when x == 0.

- Formula:

$$
z_i =
\begin{cases}
0,                                & x_i = 0 \\
x_i \cdot \log(1 + y_i),          & x_i \neq 0
\end{cases}
$$

- x and y support broadcasting, and the output shape is the maximum shape after broadcasting.
- If y is NaN, the output is the original value of y.

## Function Prototype

Each operator uses a [two-phase APIs](../../../docs/en/context/two_phase_api.md). Call "aclnnXlog1pyGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnXlog1py" to perform the computation.

```Cpp
aclnnStatus aclnnXlog1pyGetWorkspaceSize(
  const aclTensor  *x,
  const aclTensor  *y,
  const aclTensor  *z,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnXlog1py(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  aclrtStream        stream)
```

## aclnnXlog1pyGetWorkspaceSize

- **Parameters:**

  <table style="table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>x</td>
      <td>Input</td>
      <td>Multiplier factor, corresponding to x in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must be the same as that of y and z.</li><li>The shape must satisfy the broadcast relationship with y.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Independent variable of log1p, corresponding to y in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must be the same as that of x.</li><li>The shape must satisfy the broadcast relationship with x.</li></ul></td>
      <td>The data type must be the same as that of x.</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Represents the calculation result, corresponding to z in the formula.</td>
      <td>The shape must be the broadcast shape of x and y.</td>
      <td>The data type must be the same as that of x.</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the Device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input tensor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or data format of x, y, or z is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of x, y, and z do not match.</td>
    </tr>
    <tr>
      <td>The shape dimension of x, y, or z is not within the supported range (a maximum of 8 dimensions).</td>
    </tr>
    <tr>
      <td>x, y, or z uses a private format.</td>
    </tr>
  </tbody></table>

## aclnnXlog1py

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnXlog1pyGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnXlog1py defaults to a deterministic implementation.
- The input data type must be FLOAT, FLOAT16, or BFLOAT16.
- The shape dimension ranges from 0 to 8.
- Private format is not supported; only ND format is supported.
- The data types of inputs x and y and output z must be the same.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_xlog1py.h"

#define CHECK_RET(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("[FAIL] " msg "\n"); \
            return -1; \
        } \
    } while (0)

#define LOG_PRINT(msg, ...) printf(msg "\n", ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto i : shape) size *= i;
    return size;
}

// Broadcast index: map flat index in output to flat index in input
static int64_t BroadcastIdx(int64_t flat, const std::vector<int64_t>& inShape,
                             const std::vector<int64_t>& outShape)
{
    int inRank = (int)inShape.size();
    int outRank = (int)outShape.size();
    int64_t outIdx = 0, outStride = 1;
    for (int d = 0; d < outRank; d++) {
        int dimIdx = outRank - 1 - d;
        int64_t dim = outShape[dimIdx];
        int64_t coord = (flat / outStride) % dim;
        int inDimIdx = dimIdx - (outRank - inRank);
        int64_t inDim = (inDimIdx >= 0) ? inShape[inDimIdx] : 1;
        int64_t inCoord = (inDim == 1) ? 0 : coord;
        int inStride = 1;
        for (int dd = inRank - 1; dd > inDimIdx; dd--) inStride *= inShape[dd];
        outIdx += inCoord * inStride;
        outStride *= dim;
    }
    return outIdx;
}

std::vector<float> ComputeGolden(
    const std::vector<float>& x, const std::vector<int64_t>& shapeX,
    const std::vector<float>& y, const std::vector<int64_t>& shapeY,
    const std::vector<int64_t>& outShape)
{
    int64_t n = GetShapeSize(outShape);
    std::vector<float> result(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t ix = BroadcastIdx(i, shapeX, outShape);
        int64_t iy = BroadcastIdx(i, shapeY, outShape);
        float fx = x[ix], fy = y[iy];
        if (fx == 0.0f) {
            result[i] = 0.0f;
        } else {
            result[i] = fx * std::log1p(fy);
        }
    }
    return result;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");
    return 0;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc failed");
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D failed");

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int RunXlog1py(const std::vector<int64_t>& shapeX, const std::vector<float>& dataX,
               const std::vector<int64_t>& shapeY, const std::vector<float>& dataY,
               const std::string& tag, aclrtStream stream)
{
    LOG_PRINT("--- Test %s ---", tag.c_str());

    // Compute broadcast output shape
    int rank = std::max(shapeX.size(), shapeY.size());
    std::vector<int64_t> outShape(rank);
    for (int d = 0; d < rank; d++) {
        int dx = d - (rank - shapeX.size());
        int dy = d - (rank - shapeY.size());
        auto sx = (dx >= 0) ? shapeX[dx] : 1;
        auto sy = (dy >= 0) ? shapeY[dy] : 1;
        outShape[d] = std::max(sx, sy);
    }
    int64_t outSize = GetShapeSize(outShape);

    // Compute golden
    auto golden = ComputeGolden(dataX, shapeX, dataY, shapeY, outShape);

    // Allocate device tensors
    aclTensor* aclX = nullptr; void* devX = nullptr;
    auto ret = CreateAclTensor(dataX, shapeX, &devX, aclDataType::ACL_FLOAT, &aclX);
    CHECK_RET(ret == 0, "create tensor X failed");

    aclTensor* aclY = nullptr; void* devY = nullptr;
    ret = CreateAclTensor(dataY, shapeY, &devY, aclDataType::ACL_FLOAT, &aclY);
    CHECK_RET(ret == 0, "create tensor Y failed");

    std::vector<float> outHostData(outSize, 0);
    aclTensor* aclOut = nullptr; void* devOut = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &devOut, aclDataType::ACL_FLOAT, &aclOut);
    CHECK_RET(ret == 0, "create tensor Out failed");

    // Phase 1: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnXlog1pyGetWorkspaceSize(aclX, aclY, aclOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1pyGetWorkspaceSize failed");

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed");
    }

    // Phase 2: Execute
    ret = aclnnXlog1py(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnXlog1py execute failed");

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");

    // Copy result back and compare
    std::vector<float> npuResult(outSize, 0);
    ret = aclrtMemcpy(npuResult.data(), outSize * sizeof(float), devOut,
                      outSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy result D2H failed");

    bool pass = true;
    for (int64_t i = 0; i < outSize; i++) {
        float g = golden[i];
        float r = npuResult[i];
        double mere = (std::fabs(g) > 1e-6) ?
            std::fabs(r - g) / std::fabs(g) : std::fabs(r - g);
        if (mere > 0.001) {
            LOG_PRINT("  [FAIL][%s][%ld] golden=%.6f npu=%.6f", tag.c_str(), i, g, r);
            pass = false;
        }
    }
    if (pass) LOG_PRINT("  [PASS][%s] all %ld elems OK", tag.c_str(), outSize);

    // Cleanup
    aclDestroyTensor(aclX); aclDestroyTensor(aclY); aclDestroyTensor(aclOut);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devOut);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    return pass ? 0 : -1;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, "Init failed");

    int numPass = 0, numFail = 0;

    // Test 1: same shape [1,2,4,4]
    {
        std::vector<int64_t> shape = {1, 2, 4, 4};
        std::vector<float> x(32), y(32);
        for (int i = 0; i < 32; i++) { x[i] = 2.0f; y[i] = 1.0f; }
        if (RunXlog1py(shape, x, shape, y, "same_shape", stream) == 0) numPass++; else numFail++;
    }

    // Test 2: broadcast x=[1,2,1,4]  y=[1,2,4,4]
    {
        std::vector<int64_t> shapeX = {1, 2, 1, 4};
        std::vector<int64_t> shapeY = {1, 2, 4, 4};
        std::vector<float> x(8), y(32);
        for (int i = 0; i < 8; i++)  x[i] = 3.0f;
        for (int i = 0; i < 32; i++) y[i] = 2.0f;
        if (RunXlog1py(shapeX, x, shapeY, y, "broadcast", stream) == 0) numPass++; else numFail++;
    }

    // Test 3: x == 0 boundary case
    {
        std::vector<int64_t> shape = {1, 1, 8, 8};
        std::vector<float> x(64, 0.0f);
        std::vector<float> y(64, 100.0f);
        if (RunXlog1py(shape, x, shape, y, "x_eq_0", stream) == 0) numPass++; else numFail++;
    }

    LOG_PRINT("========================================");
    LOG_PRINT("ACLNN Xlog1py NPU results: PASS=%d  FAIL=%d", numPass, numFail);
    LOG_PRINT("========================================");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (numFail == 0) ? 0 : -1;
}
```
