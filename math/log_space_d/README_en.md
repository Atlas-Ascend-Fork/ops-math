# LogSpaceD

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

LogSpaceD receives the auxiliary tensor `assist` with a fixed shape and generates a logarithmically spaced sequence of length `steps` according to the TBE single-ended FP32 semantics. `assist` is fixed to rank-1 `[1]` and only serves as the ABI input slot and the dtype profile. Its values do not participate in the computation.

Dynamic capability: `DynamicRankSupportFlag=true`. During graph building, `[-2]` (unknown rank) can be declared, but the runtime does not relax the contract: `assist.shape=[1]`, the output rank is fixed to 1, and the shape is derived from attributes as `[steps]`.

Let `n=steps`. If `n>1`, `ratio=(end-start)/(n-1)`; otherwise, `ratio=end-start`. All attribute conversions, interpolation, and subsequent mathematical operations are performed in FP32:

$$
x_i=\operatorname{FP32}(i)\times ratio+\operatorname{FP32}(start),\quad i=0,1,\ldots,steps-1.
$$

The output adopts the TBE three-branch continuous semantics:

$$
y_i^{(32)}=\begin{cases}
\exp(x_i\ln(base)), & base>0,\\
\left[1-2\left(|x_i|-2\left\lfloor |x_i|/2\right\rfloor\right)\right]
\exp(x_i\ln|base|), & base<0,\\
1, & base=0\ \land\ x_i=0,\\
0, & base=0\ \land\ x_i\ne0.
\end{cases}
$$

Both `dtype=0/1` are compatibility no-ops, and the output type is always the same as that of `assist`: FP16→FP16, FP32→FP32, and BF16→BF16. When `steps=0`, the input is still `[1]` and the output is `[0]`. When `steps=1`, the global index is 0, and the result is determined only by `start`.

## Parameter Description

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>Parameter Name</th>
<th>Input/Output/Attribute</th>
<th>Description</th>
<th>Data Type</th>
<th>Data Format</th>
</tr>
</thead>
<tbody>
<tr><td>assist</td><td>Input</td><td>Fixed rank-1 [1] carrier, whose values do not participate in the computation.</td><td>FLOAT16, FLOAT, BFLOAT16</td><td>ND</td></tr>
<tr><td>y</td><td>Output</td><td>Logarithmically spaced sequence with the shape [steps]. The data type is the same as that of assist, and the dtype is a compatibility no-op.</td><td>FLOAT16, FLOAT, BFLOAT16</td><td>ND</td></tr>
<tr><td>start</td><td>Attribute</td><td>Start of the exponent domain, required.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>end</td><td>Attribute</td><td>End of the exponent domain, required.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>steps</td><td>Optional attribute</td><td>Number of output elements. The default value is 100.</td><td>INT64</td><td>-</td></tr>
<tr><td>base</td><td>Optional attribute</td><td>Base of the power. The default value is 10.0.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>dtype</td><td>Optional attribute</td><td>Historical compatibility attribute. Only 0 or 1 is allowed, and the default value is 1. Neither value changes the output type.</td><td>INT64</td><td>-</td></tr>
</tbody>
</table>

## Constraint Description

- Graph building supports `DynamicRankSupportFlag=true`, and `[-2]` unknown rank can be declared. At runtime, `assist` must be `[1]`, and `y` must be `[steps]`. Their shapes are not bound and are not reused in place.
- `assist` supports FLOAT16, FLOAT, and BFLOAT16 inputs, and the output type is the same as the input type. Together with `dtype=0/1`, they form six valid combinations.
- `dtype` allows only 0 or 1, both of which are compatibility no-ops and do not change the output type.
- FLOAT64 input, FLOAT64 output, and FLOAT64 mathematical operations are not supported. All combinations use the FP32 main computation chain, and the FLOAT16 output is converted only once at the end.
- `start`, `end`, and `base` must be finite FLOAT values. A negative base uses the continuous remainder sign branch, and a zero base uses the `x_i==0 ? 1 : 0` branch. The special value semantics of the general Pow are not used.
- `steps` must be non-negative and only determines the output length. The input length is fixed to 1. Only the ND format is supported. Broadcasting, endpoint tensors, `axis`, and `endpoint=False` are not supported.
- Only the independent GEIR invocation path of the Ascend 950PR&950DT products is supported, and the algorithm Workspace is 0. The `result` of the ACLNN combination path may be a multi-dimensional tensor. The runtime assist constraint of the independent GEIR cannot be rewritten based on it.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| GE graph mode | - | For the graph IR definition, refer to [the LogSpaceD prototype](op_graph/log_space_d_proto.h). The operator prototype is `LogSpaceD(assist, start, end, steps=100, base=10.0, dtype=1) -> y`. |
