# aclnnLinalgCholesky

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：计算一个或多个实数对称正定矩阵的Cholesky分解。
- 当upper为false时，out为下三角矩阵，满足：

  $$
  self = out \times out^{H}
  $$

- 当upper为true时，out为上三角矩阵，满足：

  $$
  self = out^{H} \times out
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnLinalgCholeskyGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLinalgCholesky”接口执行计算。

```Cpp
aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(
  const aclTensor   *self,
  bool              upper,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnLinalgCholesky(
  void               *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor      *executor,
  const aclrtStream  stream)
```

## aclnnLinalgCholeskyGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1537px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>shape的最后两维需要相等，self需为实数对称正定矩阵。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>upper（bool）</td>
      <td>输入</td>
      <td>选择cholesky上三角分解还是下三角分解。</td>
      <td><ul><li>True：实现上三角cholesky分解。</li><li>False：实现下三角cholesky分解。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td><ul><li>数据类型与self相同。</li><li>shape与self相同。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  本接口不校验 self 数据的对称性和正定性。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self、out、workspaceSize或executor为空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self和out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self和out的存储格式不一致，或使用私有格式。</td>
    </tr>
    <tr>
      <td>self的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self的维度大于8。</td>
    </tr>
    <tr>
      <td>self和out的shape不一致。</td>
    </tr>
    <tr>
      <td>self的维度小于2。</td>
    </tr>
    <tr>
      <td>self的最后两维不相等。</td>
    </tr>
    <tr>
      <td>self最后一维的大小超过8192。</td>
    </tr>
  </tbody>
  </table>

## aclnnLinalgCholesky

- **参数说明：**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLinalgCholeskyGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  对于非正定输入，计算可能失败，错误可能在同步stream时报告。

## 约束说明

- 维度约束：self的维度（rank）必须在[2, 8]范围内，最后两维必须相等构成方阵，最后一维M必须在[0, 8192]范围内；不满足时返回161002（ACLNN_ERR_PARAM_INVALID）。
- 数据类型：支持FLOAT、FLOAT16、BFLOAT16。
- 空Tensor：满足参数约束的空Tensor为合法输入，第一段接口返回成功，workspaceSize为0。
- 确定性：计算结果具有确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_linalg_cholesky.h"

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

// 守卫在提交前或 stream 同步成功后的普通退出路径尝试回收资源。
using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
void DestroyUnusedExecutor(aclOpExecutor* executor)
{
    if (executor == nullptr) {
        return;
    }
    auto ret = aclSetAclOpExecutorRepeatable(executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret);
        return;
    }
    ret = aclDestroyAclOpExecutor(executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclDestroyAclOpExecutor failed. ERROR: %d\n", ret);
    }
}

using ExecutorPtr = std::unique_ptr<aclOpExecutor, decltype(&DestroyUnusedExecutor)>;

// Standalone failure boundary: never release submitted buffers before a successful wait.
class PendingStreamGuard {
public:
    explicit PendingStreamGuard(aclrtStream stream) noexcept : stream_(stream) {}
    PendingStreamGuard(const PendingStreamGuard&) = delete;
    PendingStreamGuard& operator=(const PendingStreamGuard&) = delete;

    void MarkSubmitted() noexcept { pending_ = true; }
    void RecordApiStatus(int status) noexcept
    {
        apiStatus_ = status;
        apiStatusKnown_ = true;
    }

    aclError Synchronize(bool fromCleanup = false) noexcept
    {
        const auto syncStatus = aclrtSynchronizeStream(stream_);
        if (syncStatus != ACL_SUCCESS) {
            // A failed wait does not establish completion. This standalone process stops
            // without normal destruction; it does not claim graceful resource recovery.
            std::fprintf(stderr,
                         "Pending stream failed: api2_status_known=%d api2_status=%d "
                         "sync_status=%d cleanup=%d; exiting without normal cleanup\n",
                         static_cast<int>(apiStatusKnown_), apiStatus_, static_cast<int>(syncStatus),
                         static_cast<int>(fromCleanup));
            std::fflush(stderr);
            std::_Exit(EXIT_FAILURE);
        }
        pending_ = false;
        return syncStatus;
    }

    ~PendingStreamGuard() noexcept
    {
        if (pending_) {
            (void)Synchronize(true);
        }
    }

private:
    aclrtStream stream_;
    bool pending_ = false;
    bool apiStatusKnown_ = false;
    int apiStatus_ = 0;
};

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存，成功后立即交由 deviceAddr 守卫
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor，成功后立即交由 tensor 守卫
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int RunExample()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    // 待提交任务完成后，普通作用域退出才按置位状态尝试回收 Device 与 ACL 上下文。
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {3, 3};
    std::vector<int64_t> outShape = {3, 3};
    DeviceMemPtr selfDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr outDeviceAddr(nullptr, &aclrtFree);
    TensorPtr self(nullptr, &aclDestroyTensor);
    TensorPtr out(nullptr, &aclDestroyTensor);
    std::vector<float> selfHostData = {6, -3, 1, -3, 2, 0, 1, 0, 4};
    std::vector<float> outHostData(9, 0);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, aclDataType::ACL_FLOAT, selfDeviceAddr, self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, aclDataType::ACL_FLOAT, outDeviceAddr, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    aclOpExecutor* executor = nullptr;
    // 调用aclnnLinalgCholesky第一段接口
    ret = aclnnLinalgCholeskyGetWorkspaceSize(self.get(), false, out.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCholeskyGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ExecutorPtr executorGuard(executor, &DestroyUnusedExecutor);
    // 根据第一段接口计算出的workspaceSize申请device内存，成功后交由 workspaceAddr 守卫
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }
    // 调用aclnnLinalgCholesky第二段接口
    PendingStreamGuard pendingStream(stream.get());
    pendingStream.MarkSubmitted();
    ret = aclnnLinalgCholesky(workspaceAddr.get(), workspaceSize, executorGuard.release(), stream.get());
    pendingStream.RecordApiStatus(ret);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgCholesky failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = pendingStream.Synchronize();

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr.get(),
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 已提交任务仅在 stream 同步成功后执行普通资源析构。
    // 同步报错时本独立示例直接终止，不声称资源清理成功。
    // pending executor 清理 API 失败会记录错误，不保证回收成功。
    return 0;
}

int main()
{
    try {
        return RunExample();
    } catch (const std::bad_alloc& error) {
        std::fprintf(stderr, "Host allocation failed: %s\n", error.what());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Host exception: %s\n", error.what());
    }
    return EXIT_FAILURE;
}
```
