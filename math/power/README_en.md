# Power

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term>      |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs a linear transformation on each element of the input tensor `x` and then performs the power computation.

- Computation formula:

$$
y_i = \big(\text{scale} \cdot x_i + \text{shift}\big)^{\text{power}}
$$

  where `power` / `scale` / `shift` are all scalar attributes. All branch decisions and precomputable constant folding are completed in the tiling phase on the host side, and the kernel side routes to the corresponding DAG based on `tilingKey` without runtime branches.

- Equivalent PyTorch expression:

  ```python
  base = scale * x + shift
  y = torch.pow(base, power)
  ```

- Implementation notes:

  | Category | Content |
  |---|---|
  | Computation template | Elementwise (`ElewiseBaseTiling` + `ElementwiseSch` + `DAGSch`) |
  | Computation precision | `fp16`/`bf16` are cast to `fp32` in the kernel for computation and then written back to the original dtype |
  | Branch hoisting | `culType` × `dtype` × `schMode` are encoded into `tilingKey` in three segments, and the DAG has been selected when the kernel template is instantiated |
  | Performance optimization | `power ∈ {1,2,3}` uses multiplication expansion; `power ∉ {0,1,2,3}` uses the general path of `exp(power·log(|base|))` |

- Internal operator branches (`culTypeEnum` on the host side):

  | culType | Trigger Condition | Computation |
  |---|---|---|
  | `ALL_ZEROS` | `scale·power == 0` and `shift==0` and `power>0` | `y = 0` |
  | `BROADCAST_SCALAR` | `power==0`, or `scale==0` and `power≠0` | `y = bcastVal` (the host side precomputes one of `pow(shift,power)`, `1.0`, `NaN`, and `+inf`) |
  | `LINEAR` | `scale·power≠0` and `power==1` | `y = x·scale + shift` |
  | `SQUARE` | `scale·power≠0` and `power==2` | `y = (x·scale + shift)^2` |
  | `CUBE` | `scale·power≠0` and `power==3` | `y = (x·scale + shift)^3` |
  | `GENERIC_POW_POS` | `scale·power≠0` and `power>0` and `power∉{1,2,3}` | General power computation. Outputs `0` when `base==0` |
  | `GENERIC_POW_NEG` | `scale·power≠0` and `power<0` | General power computation. Outputs `+inf` when `base==0` |

  For `power∈{0,1,2,3}`, the `IsClose` equality tolerance is `atol=1e-8` and `rtol=1e-5`, aligned with `math/is_close`.

## Operator Parameter Description

  <table style="table-layout: fixed; width: 1280px"><colgroup>
  <col style="width: 140px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Instructions</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (Tensor)</td>
      <td>Input</td>
      <td>Base pre-transform quantity for the power computation, which is x in the formula.</td>
      <td>The shape is exactly the same as that of y; broadcasting is not supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>No limit (including the 0-dimension scalar; a scalar is processed as [1])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y (Tensor)</td>
      <td>Output</td>
      <td>Power computation result, which is y in the formula.</td>
      <td>The dtype is the same as that of x, and the shape is the same as that of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>Same as x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>power (attr, float)</td>
      <td>Attribute</td>
      <td>Power exponent, which is power in the formula.</td>
      <td>Optional, default <code>1.0</code>.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>Scalar</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (attr, float)</td>
      <td>Attribute</td>
      <td>Linear scaling factor of the input, which is scale in the formula.</td>
      <td>Optional, default <code>1.0</code>.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>Scalar</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift (attr, float)</td>
      <td>Attribute</td>
      <td>Linear shift amount of the input, which is shift in the formula.</td>
      <td>Optional, default <code>0.0</code>.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>Scalar</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Special Value Conventions

Each element follows the following table (`base = scale·x + shift`; an integer power means `power == floor(power)` and is finite):

  | Scenario | Output Value | Description |
  |---|---|---|
  | `base > 0` | `exp(power · log(base))` | General main path |
  | `base < 0` and `power` is an integer | `(-1)^power · exp(power · log(|base|))` | The host presets `negScalar = ±1`, and the kernel side merges the results using `Compare + Select` |
  | `base < 0` and `power` is not an integer | `NaN` | Undefined in the real number domain. The host presets `negScalar = NaN`, and the multiply-add propagation yields NaN |
  | `base == 0` and `power > 0` | `0` | GENERIC_POW_POS / ALL_ZEROS |
  | `base == 0` and `power < 0` | `+inf` | GENERIC_POW_NEG / BROADCAST_SCALAR, IEEE 754 semantics |
  | `power == 0` (including `0^0`) | `1` | By convention, the BROADCAST_SCALAR path is used, and the host presets `bcastVal = 1.0` |

## Constraint Description

- **Data type**: The input `x` and output `y` must be one of `FLOAT16` / `BFLOAT16` / `FLOAT`, and the two must be exactly the same. The tiling on the host side rejects other dtypes and returns `GRAPH_FAILED`.
- **Same shape**: Broadcasting is not supported. The shapes of `x` and `y` must be exactly the same. A 0-dimension scalar is regarded as `[1]` on the host side.
- **Platform**: Currently, only the kernel binary for the `ascend950` platform is generated. When compiling other platforms, this operator is not delivered.
- **API form**: This operator does **not** provide an aclnn single-operator API. It exists only as a graph operator node and must be integrated through GE IR or a graph compiler. For host direct invocation (kernel-launch) scenarios, directly reuse the kernel source code in this repository instead of the aclnn APIs.
- **Attribute default values**: By default, `power=1.0`, `scale=1.0`, and `shift=0.0`, which is equivalent to the identity mapping (`y = x`). In this case, the `LINEAR` path is used.
- **Deterministic computation**: This operator uses a deterministic implementation by default. The outputs are the same for each execution with the same inputs and attributes.

## Invocation Method

Invoking this operator through the aclnn API is not yet supported. The recommended integration method:

 **Graph operator method**: When building a graph through GE IR / ATC, add `Power` as a node to the computation graph, and the graph compiler automatically completes tiling, kernel selection, and delivery. The attributes `power` / `scale` / `shift` are injected through node attrs.
