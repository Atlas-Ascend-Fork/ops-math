# ReduceStdV2Update

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

> For the mapping between products and chips, refer to [Ascend Product Form Description](https://www.hiascend.com/document/redirect/CannCommunityProductForm). <term>Ascend 950PR&950DT products</term> is the target platform of the newly added Ascend C Kernel implementation in this update. The other supported products are covered by the existing TBE binary.

## Function Description

- Operator function: receives the raw input `x` and the pre-computed mean `mean` (already broadcast to the shape of `x`), computes the sum of squared deviations along the specified dimension `dim` by reduction, and then divides the sum by `N - correction` to obtain the variance or standard deviation. Bessel correction (correction) and keepdim are supported. ReduceStdV2Update is an internal operator in CANN l0op. It is internally called by `aclnnVar`/`aclnnVarCorrection`/`aclnnVarMean` on the non-regbase path, so that `aclnnVarMean` can reuse the mean result to avoid computing the mean repeatedly.
- Computation formula:

  $$
  \text{diff} = x - \mu
  $$

  $$
  \text{sq\_sum} = \sum_{i \in D} \text{diff}_{i}^{2}
  $$

  $$
  \text{var} = \frac{\text{sq\_sum}}{N - \text{correction}}
  $$

  $$
  \text{output} = \begin{cases} \sqrt{\text{var}} & \text{if } \text{if\_std} = \text{true} \\ \text{var} & \text{if } \text{if\_std} = \text{false} \end{cases}
  $$

  Here, $D$ is the set of reduction dimensions, $N$ is the number of elements in the reduction dimensions, and $\mu$ is the mean passed in externally. correction=0 indicates a biased estimate (denominator N), and correction=1 indicates an unbiased estimate (denominator N-1).

- Differences between APIs: this operator provides two equivalent L0 APIs. `ReduceStdV2Update` uses the `bool unbiased` parameter (true=unbiased N-1, false=biased N) and is called by `aclnnVar`. `ReduceStdV2UpdateCorrection` uses the `int64_t correction` parameter (0=biased, 1=unbiased) and is called by `aclnnVarCorrection`. The two APIs are mathematically equivalent and differ only in the parameter expression. The L0 API always outputs the variance (if_std=false). In GE IR graph mode, the `if_std` attribute can be used to control whether to output the variance or the standard deviation.

## Parameter Description

<table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 280px">
  <col style="width: 150px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (self)</td>
      <td>Input</td>
      <td>Raw input data, corresponding to x in the formula. 0-8 dimensional ND.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Pre-computed mean, corresponding to μ in the formula. It must have been broadcast to the shape of self through Expand. Its dtype is the same as that of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output (output_var)</td>
      <td>Output</td>
      <td>Variance or standard deviation, corresponding to output in the formula. Its dtype is the same as that of x, and its shape is the shape of x after reduction along dim.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>List of reduction dimensions, corresponding to D in the formula. Negative indexes are supported. If dim is empty, all dimensions are reduced.</td>
      <td>INT64 (ListInt)</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>if_std</td>
      <td>Attribute</td>
      <td>false=output variance, true=output standard deviation (square root). The default value is false. The value is fixed to false for the aclnn API.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unbiased</td>
      <td>Attribute</td>
      <td>Specifies whether to use Bessel correction. true=uses correction for the denominator, false=forces biased (denominator N). The default value is true. Mutually exclusive with correction.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>Attribute</td>
      <td>Specifies whether to retain the reduction dimensions. true=set the dimension to 1, false=remove the dimension. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>correction</td>
      <td>Attribute</td>
      <td>Bessel correction factor, any non-negative integer (aligned with the torch.var value domain). The denominator is N-correction, and NaN is output when N<=correction. The default value is 1. Mutually exclusive with unbiased.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraint Description

- Input data type restriction: `x` and `mean` support only FLOAT, FLOAT16, and BFLOAT16, and their data types must be the same. DOUBLE, complex, and integer types are not supported.
- shape restriction: `mean` must have been broadcast to the shape of `x` through Expand. The output shape is the shape of `x` after reduction along `dim` (the dimension is set to 1 if keepdim=true, and is removed if keepdim=false).
- correction/unbiased restriction: correction supports any non-negative integer (denominator N-correction; NaN when N<=correction, consistent with the baseline and the torch.var routing), and negative values are rejected. unbiased and correction are mutually exclusive: unbiased=false forces biased (denominator N, correction is ignored); unbiased=true uses correction for the denominator. Setting both at the same time is not recommended.
- Edge cases: scenarios with an empty tensor, or a single element and correction≥1, are usually handled in advance by the upper-layer `aclnnVar`/`aclnnVarCorrection`. The kernel still performs defensive processing. In the EMPTY_A path where the output is empty, no operation is performed. In the EMPTY_R path where the reduction axis is empty but the output is not empty, NaN is returned. In the normal path, when `N <= correction`, the same semantics are maintained through `cof=NaN`.
- Precision restrictions (the 950 implementation is strictly aligned with the canndev prototype):
  - FP16/BF16 inputs are always promoted to FP32 for accumulation (fixed promotion, without GetPromoteType). FP32 inputs are computed directly. The results are converted back to the original dtype.
  - The conversion back to BF16 output uses the round mode (corresponding to AscendC Cast round), retaining the rounding semantics of BF16.
  - When if_std=true, high_precision vsqrt is used for the square root operation.
  - The output dtype is the same as that of the input x, and no type promotion is performed at the API layer.
- Input size restriction: the GM stride of any axis of `x`/`mean` (the element count of all dimensions after
  that axis multiplied by the dtype size in bytes) must not exceed the uint32 limit (about 4GiB); when the stride
  consumed by block-to-block copy exceeds the limit, it is explicitly rejected at the Tiling stage. Boundary
  tensors with 2^31 or more elements can still be computed normally as long as the axis strides stay within the
  limit.
- Dynamic shape support: dynamic compilation, dynamic shape, and dynamic rank are registered. When building a graph, use -1 for an unknown dimension and -2 for an unknown rank; they are concretized with the actual shape at RunGraph (the preceding dtype/shape/rank constraints must be satisfied). With keepdim=false, the output rank shrinks with the input rank.
- Determinism: this operator is deterministic by default.

## Invocation Description

| Invocation Method | Sample Code | Description |
|---------|---------|------|
| GE graph-mode invocation | [test_geir_reduce_std_v2_update](./examples/arch35/test_geir_reduce_std_v2_update.cpp) | Invokes the ReduceStdV2Update operator by building a graph through the [operator IR](./op_graph/reduce_std_v2_update_proto.h). The if_std attribute is supported to control whether to output the variance or the standard deviation. |

> Note: this operator is an internal sub-operator of l0op and does not expose an independent aclnn API externally. The L0 API always outputs the variance with if_std=false. In GE IR graph mode, the if_std attribute can be used to control whether to output the variance or the standard deviation. The upper-layer `aclnnVar`/`aclnnVarCorrection`/`aclnnVarMean` internally call this operator on the non-regbase path.
