# aclnnSinkhorn

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

- 接口功能：计算Sinkhorn最优传输计划，可以用于MoE模型中的专家路由。

- 计算公式：

  $$
  p=Sinkhorn(cost, tol)
  $$

  **输入**：

  - cost(R, C)：2维成本矩阵
  - tol：误差

  **初始化**：

  $$
  cost = exp(cost) \\
  d0 = ones(R) \\
  d1 = ones(C) \\
  eps = 0.00000001 \\
  error = 1e9 \\
  d1_old = d1
  $$

  **重复执行**：

  $$
  d0 = \frac{1}{R * (sum(d1 * cost, 1) + eps)} \\
  d1 = \frac{1}{C * (sum(d0.unsqueeze(1) * cost, 0) + eps)} \\
  error = mean(abs(d1_old - d1)) \\
  d1_old = d1
  $$

  直至满足：

  $$
  error <= tol
  $$

  **输出**：

  $$
  p = d1 * cost * d0.unsqueeze(1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnSinkhornGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSinkhorn"接口执行计算。

```cpp
aclnnStatus aclnnSinkhornGetWorkspaceSize(
  const aclTensor  *cost,
  const aclScalar  *tol,
  aclTensor        *p,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSinkhorn(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnSinkhornGetWorkspaceSize

- **参数说明**：

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
      <td>cost（const aclTensor*）</td>
      <td>输入</td>
      <td>表示成本张量，公式中的<code>cost</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>取值建议在[0,1]之间；域外仍参与计算，可能产生Inf或NaN。</li><li>二维矩阵，行数无固定上限，列数不超过4096，元素总数不超过UINT32_MAX。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tol（const aclScalar*）</td>
      <td>可选输入</td>
      <td>表示计算Sinkhorn的误差，公式中的<code>tol</code>。</td>
      <td><ul><li>传入空指针时使用默认值0.0001。</li><li>支持全部FLOAT取值；NaN、±Inf和负值按浮点比较语义参与终止判断。</li></ul></td>
      <td>可转换为FLOAT的标量类型</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>p（aclTensor*）</td>
      <td>输出</td>
      <td>表示最优传输张量，公式中的<code>p</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与cost一致。</li><li>cost为FLOAT时p必须为FLOAT且连续；cost为FLOAT16/BFLOAT16时p支持FLOAT、FLOAT16或BFLOAT16。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√<sup>1</sup></td>
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

  > <sup>1</sup> cost为非空FLOAT时不支持非连续p；cost为FLOAT16/BFLOAT16或空Tensor时支持合法非连续p。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 301px">
  <col style="width: 137px">
  <col style="width: 714px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的cost、p、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>cost和p的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cost为FLOAT，但p不是FLOAT。</td>
    </tr>
    <tr>
      <td>cost的shape不是二维，cost的维度为负、列数超过4096或元素总数超过UINT32_MAX。</td>
    </tr>
    <tr>
      <td>cost和p的shape不一致。</td>
    </tr>
    <tr>
      <td>非空cost为FLOAT且p为非连续Tensor。</td>
    </tr>
    <tr>
      <td>cost或p的数据格式不是ND。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561000</td>
      <td>发生内部错误，请携带完整日志联系技术支持。</td>
    </tr>
  </tbody>
  </table>

## aclnnSinkhorn

- **参数说明**：

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSinkhornGetWorkspaceSize获取。</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnSinkhorn默认确定性实现。

- 收敛保护：内部迭代最多执行10000次。若10000次内误差仍未收敛至`tol`，计算将停止并以第10000次迭代的状态作为输出结果正常结束。

- 特殊值终止：严格使用`!(error > tol)`；NaN比较会终止，`+Inf`与有限tol比较时继续迭代，最终仍受10000次上限保护。

- 空Tensor：R或C为0时返回成功且workspaceSize为0。

- 数值范围：`[0,1]`仅为建议域，域外值仍参与计算，可能产生Inf或NaN。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                  \
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

int Init(int32_t deviceId, aclrtStream *stream) {
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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                             shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Device/Stream初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  std::vector<int64_t> costShape = {3, 2};
  std::vector<int64_t> pShape = {3, 2};
  void* costDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  aclTensor* cost = nullptr;
  aclScalar* tol = nullptr;
  aclTensor* p = nullptr;
  std::vector<float> costHostData = {0.45f, 0.48f, 0.65f, 0.68f, 0.68f, 0.10f};
  std::vector<float> pHostData(6, 0);

  float tolValue = 0.0001;

  ret = CreateAclTensor(costHostData, costShape, &costDeviceAddr, aclDataType::ACL_FLOAT, &cost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  tol = aclCreateScalar(&tolValue, aclDataType::ACL_FLOAT);
  CHECK_RET(tol != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return -1);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSinkhornGetWorkspaceSize(cost, tol, p, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhornGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSinkhorn(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSinkhorn failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出值
  auto size = GetShapeSize(pShape);
  std::vector<float> pData(size, 0);
  ret = aclrtMemcpy(pData.data(), pData.size() * sizeof(pData[0]), pDeviceAddr,
                    size * sizeof(pData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("p result[%ld] is: %e\n", i, pData[i]);
  }

  // 6. 释放资源
  aclDestroyTensor(cost);
  aclDestroyTensor(p);
  aclDestroyScalar(tol);

  aclrtFree(costDeviceAddr);
  aclrtFree(pDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
