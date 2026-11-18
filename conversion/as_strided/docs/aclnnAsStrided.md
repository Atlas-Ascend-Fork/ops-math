# aclnnAsStrided

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：按输出形状size、读取步幅stride和存储偏移storageOffset，从输入张量x中读取元素，
  物化生成形状为size的输出张量y。
- 计算公式：

  $$
  y_i = x_{\text{storageOffset}+\sum_{d=0}^{D-1}(i_d \cdot \text{stride}[d])}
  $$

  其中$i_d$为输出在第$d$维的下标，storageOffset为相对x视图起点的存储偏移。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAsStridedGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAsStrided”接口执行计算。

```cpp
aclnnStatus aclnnAsStridedGetWorkspaceSize(
    const aclTensor*   x,
    const aclIntArray* size,
    const aclIntArray* stride,
    int64_t            storageOffset,
    aclTensor*         y,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAsStrided(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    const aclrtStream stream)
```

## aclnnAsStridedGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px">
  <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 100px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>源数据张量，对应计算公式中的x。</td>
      <td><ul>
        <li>支持空Tensor，仅当输出同时为空时才有意义。</li>
      </ul></td>
      <td>INT64、UINT64、INT32、UINT32、FLOAT、FLOAT16、INT8、UINT8、BFLOAT16、INT16、UINT16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8（仅存储大小参与边界校验，视图shape不参与gather计算）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>size（const aclIntArray*）</td>
      <td>输入</td>
      <td>输出张量的形状，对应计算公式中的D与各维长度。</td>
      <td><ul>
        <li>长度为1~8。</li>
        <li>元素非负，取值为0表示输出为空Tensor。</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride（const aclIntArray*）</td>
      <td>输入</td>
      <td>读取x时各维的步幅，对应计算公式中的stride[d]。</td>
      <td><ul>
        <li>长度必须与size一致。</li>
        <li>元素非负，取值为0表示广播拷贝。</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>storageOffset（int64_t）</td>
      <td>输入</td>
      <td>读取x的起始存储偏移，对应计算公式中的storageOffset。</td>
      <td><ul>
        <li>非负，相对x的视图起点。</li>
      </ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量，对应计算公式中的y。</td>
      <td><ul>
        <li>支持空Tensor（size中含0时）。</li>
        <li>shape必须等于size。</li>
      </ul></td>
      <td>与x保持一致。</td>
      <td>ND</td>
      <td>1-8</td>
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
  </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="table-layout: fixed; width: 1000px">
    <colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x、y、size、stride、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>当前芯片架构不支持本接口。</td>
    </tr>
    <tr>
      <td>x或y的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>y与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>size与stride长度不一致，或长度超出[1, 8]范围。</td>
    </tr>
    <tr>
      <td>size、stride存在负值，或storageOffset为负。</td>
    </tr>
    <tr>
      <td>y的shape与size不一致。</td>
    </tr>
    <tr>
      <td>storageOffset与size、stride确定的读取范围超出x的可读存储。</td>
    </tr>
    </tbody>
    </table>

## aclnnAsStrided

- **参数说明**

    <table style="table-layout: fixed; width: 1000px">
    <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
    </colgroup>
    <thead>
            <tr>
              <th>参数名</th>
              <th>输入/输出</th>
              <th>描述</th></tr>
        </thead>
        <tbody>
            <tr>
              <td>workspace</td>
              <td>输入</td>
              <td>在Device侧申请的workspace内存地址。</td></tr>
            <tr>
              <td>workspaceSize</td>
              <td>输入</td>
              <td>在Device侧申请的workspace大小，由第一段接口aclnnAsStridedGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnAsStrided默认确定性实现。

- 由storageOffset、size、stride确定的所有读取位置必须落在x的有效存储范围内，不允许越界读取。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_as_strided.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using IntArrayPtr = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

void PrintOutResult(std::vector<int64_t>& shape, DeviceMemPtr& deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，初始化
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
    // 调用aclrtMalloc申请device侧内存
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

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API文档
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
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
    // x 为源数据张量；按 size={2,3}、stride={4,2}、storageOffset=1 物化生成输出，
    // 即 y[i0,i1] = x[1 + i0*4 + i1*2]
    std::vector<int64_t> xShape = {16};
    std::vector<int64_t> yShape = {2, 3};
    std::vector<int64_t> sizeValue = {2, 3};
    std::vector<int64_t> strideValue = {4, 2};
    int64_t storageOffset = 1;

    DeviceMemPtr xDeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr yDeviceAddr(nullptr, &aclrtFree);

    TensorPtr x(nullptr, &aclDestroyTensor);
    TensorPtr y(nullptr, &aclDestroyTensor);

    IntArrayPtr size(nullptr, &aclDestroyIntArray);
    IntArrayPtr stride(nullptr, &aclDestroyIntArray);

    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> yHostData(GetShapeSize(yShape), 0);

    ret = CreateAclTensor(xHostData, xShape, aclDataType::ACL_FLOAT, xDeviceAddr, x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, aclDataType::ACL_FLOAT, yDeviceAddr, y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建size/stride aclIntArray
    size.reset(aclCreateIntArray(sizeValue.data(), sizeValue.size()));
    CHECK_RET(size != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_FAILURE);
    stride.reset(aclCreateIntArray(strideValue.data(), strideValue.size()));
    CHECK_RET(stride != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_FAILURE);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnAsStrided第一段接口
    ret = aclnnAsStridedGetWorkspaceSize(x.get(), size.get(), stride.get(), storageOffset, y.get(), &workspaceSize,
                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnAsStridedGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
                  return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }

    // 调用aclnnAsStrided第二段接口
    ret = aclnnAsStrided(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnAsStrided failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg());
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(yShape, yDeviceAddr);

    return 0;
}
```
