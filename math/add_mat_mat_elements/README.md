# AddMatMatElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入张量`a`、`b`进行逐元素相乘后，与输入张量`c`按`alpha`、`beta`加权求和，结果写入输出张量。
- 计算公式：

  $$
  out_i = c_i \times \beta + \alpha \times a_i \times b_i
  $$

- 输出shape固定为`c.shape`。`a`、`b`分别按照尾部对齐规则广播到`c.shape`，不支持通过统一广播扩大`c`的shape。
- `alpha`、`beta`是shape为`(1,)`的Tensor，不是属性。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>c</td>
    <td>输入</td>
    <td>输出shape的锚点，按beta缩放后参与累加，rank为1~8。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>a</td>
    <td>输入</td>
    <td>参与逐元素相乘，rank取值范围为0~c的rank，shape必须能够广播到c.shape；rank为0时表示标量Tensor。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>输入</td>
    <td>参与逐元素相乘，rank取值范围为0~c的rank，shape必须能够广播到c.shape；rank为0时表示标量Tensor。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>对c进行缩放的系数Tensor，shape必须为(1,)。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>输入</td>
    <td>对a × b的乘积进行缩放的系数Tensor，shape必须为(1,)。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>c</td>
    <td>输出</td>
    <td>计算结果，shape等于输入c.shape。</td>
    <td>同c</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- `c`、`a`、`b`、`beta`、`alpha`及输出必须具有完全一致的数据类型，仅支持`FLOAT16`和`FLOAT`，不支持数据类型提升。
- `c`的rank取值范围为1~8；`a`、`b`的rank取值范围为0~`c`的rank，rank为0时按标量Tensor广播。
- `a`、`b`分别尾部对齐到`c.shape`；补齐后的每个维度必须为1或等于`c`的对应维度。
- 输出shape固定为`c.shape`。例如`c.shape=(1, 3)`、`a.shape=(2, 3)`不受支持，即使三输入能够得到统一广播shape`(2, 3)`。
- `alpha`、`beta`的shape必须严格为`(1,)`。
- 支持空Tensor，空Tensor场景仍需满足上述rank和广播约束。
- 支持动态shape和动态rank。形状推导时，`-1`表示未知维度：同一尾部对齐轴存在具体维度时使用具体维度，全部为`-1`时保留`-1`；`c`、`a`、`b`任一输入为未知rank `[-2]`时，输出为未知rank `[-2]`。运行时实际shape必须满足上述约束。
- Kernel输入输出仅支持ND格式。GE图可为部分非ND数据Tensor或请求的输出布局插入格式转换，转换为ND后再进入Kernel；格式转换能力取决于Tensor的rank和shape，不承诺任意格式组合均可转换。`alpha`、`beta`仍须使用ND格式的`(1,)` Tensor。

> 说明：当前InferShape实现按`broadcast(a, b, c)`进行形状推导，而Host Tiling以`c.shape`为输出锚点并校验`a`、`b`到`c.shape`的广播关系。实际执行约束以Host Tiling为准。

### 异常校验

| 违反条件 | 校验阶段 | 错误表现 |
| :------- | :------- | :------- |
| 缺少必选输入或输出 | 图构建/编译参数校验 | `E80001` |
| `c`的rank不在1~8范围内 | 图编译参数校验可能提前拒绝；Host Tiling保留完整范围校验 | rank大于8时当前集成通路呈现`E80012`；若进入Host Tiling则返回`GRAPH_FAILED`，当前集成通路呈现`EZ9999`并记录rank越界日志 |
| 输入dtype不支持或输入dtype不一致 | Host Tiling | 返回`GRAPH_FAILED`；当前集成通路对外呈现`EZ9999`并记录dtype校验日志 |
| `a`或`b`的rank大于`c`的rank、不能广播到`c.shape` | Host Tiling | 返回`GRAPH_FAILED`；当前集成通路对外呈现`EZ9999`并记录rank或广播校验日志 |
| `alpha`或`beta`的shape不为`(1,)` | Host Tiling | 返回`GRAPH_FAILED`；当前集成通路对外呈现`EZ9999`并记录shape校验日志 |
| 图输入声明为非ND格式 | GE格式选择/TransData | 可转换的数据Tensor由GE转换为ND后执行；不可转换的rank/shape/format组合在图编译阶段失败，不进入本算子Kernel |
| 调用方预填的输出dtype或shape与输入不一致 | 图推导 | 输出dtype跟随输入`c`，输出shape由输入shape推导，调用方预填值会被推导结果覆盖 |
| 图请求非ND输出布局 | GE格式选择/TransData | GE可在Kernel的ND输出之后执行布局转换；物理输出缓冲不能按ND裸数组直接解释 |

> 说明：错误码由图编译或运行框架对校验失败进行封装，不同集成通路可能只返回`GRAPH_FAILED`并保留具体日志。直接Kernel测试通路不负责验证输出描述符和非ND格式；此类场景应在图构建边界验证，不能以直接Kernel通路未拒绝判定为算子支持或缺陷。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| 图模式 | [test_geir_add_mat_mat_elements.cpp](examples/test_geir_add_mat_mat_elements.cpp) | 通过[算子IR](op_graph/add_mat_mat_elements_proto.h)构图方式调用AddMatMatElements算子。 |

### 图模式显式Cast说明

- 支持将显式Cast算子的输出连接至AddMatMatElements。进入AddMatMatElements时，`c`、`a`、`b`、`beta`、`alpha`必须已转换为完全一致的`FLOAT16`或`FLOAT`数据类型，`beta`、`alpha`的shape仍须为`(1,)`。
- AddMatMatElements不会为混合数据类型输入自动插入Cast，不支持隐式数据类型提升。显式Cast后输入数据类型仍不一致，或Cast目标数据类型不在`FLOAT16`、`FLOAT`范围内时，图编译失败。
- 不承诺Cast与AddMatMatElements融合，Cast可以作为独立图节点和独立Kernel执行。
