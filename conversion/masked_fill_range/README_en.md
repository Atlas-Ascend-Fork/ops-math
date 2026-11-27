# MaskedFillRange

## Product Support

| Product | Supported |
| :-- | :--: |
| <term>Ascend 950PR&950DT series products</term> | √ |
| <term>Atlas A3 series products</term> | × |
| <term>Atlas A2 series products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products</term> | × |

## Function Description

- **Operator function:** Fills the selected interval of input tensor `x` along `axis` with the value for the current round from `value`. The first dimension of `start`, `end`, and `value` is the number of fill rounds, `rounds`. Rounds are applied in order, so later rounds overwrite earlier ones.

- **Formula:** For each round `i = 0 ... rounds - 1`, `y` is initialized to `x`, and `axis` is normalized to `a ∈ [0, dim)`:

  - When the fill dimension is the last dimension `D` (1-D, 2-D, or 3-D with `a = 2`), windows are indexed by batch `n`. For 1-D, each round uses only window 0 (`s = start[i,0]`). For 2-D, row `r` is the batch (`s = start[i,r]`). For 3-D with `a = 2`, all `C` rows in batch `n` share the window (`s = start[i,n]`, `e = end[i,n]`); the interval is clamped to `[0, D)`:

    $$
    y[..., k] =
    \begin{cases}
    value[i], & \max(s, 0) \le k < \min(e, D)\\
    y[..., k], & \text{otherwise}
    \end{cases}
    $$

  - For 3-D with `a = 0`, the window is indexed by `n` (`s = start[i,n]`, `e = end[i,n]`) and compared directly without clamping. The matching plane is filled:

    $$
    y[n, :, :] =
    \begin{cases}
    value[i], & s \le n < e\\
    y[n, :, :], & \text{otherwise}
    \end{cases}
    $$

  - For 3-D with `a = 1`, the window is indexed by `n` (`s = start[i,n]`, `e = end[i,n]`) and compared directly without clamping. The matching row is filled:

    $$
    y[n, c, :] =
    \begin{cases}
    value[i], & s \le c < e\\
    y[n, c, :], & \text{otherwise}
    \end{cases}
    $$

  - For 1-D, each round uses only window 0 (`start[i,0]`/`end[i,0]`). Any additional columns in the second dimension of `start` and `end` are ignored.

## Parameters

| Name | Input/Output/Attribute | Description | Data Type | Format |
| :-- | :-- | :-- | :-- | :-- |
| `x` | Input | Input tensor in the formula. Supports 1-D `(D,)`, 2-D `(N, D)`, and 3-D `(N, C, D)`. | FLOAT16, FLOAT, INT8, INT32 | ND |
| `start` | Input | Start positions of fill intervals. Shape `(rounds, B)`: the first dimension is the number of rounds and the second is the number of window batches. Windows are indexed by row/batch; see Function Description. | INT32 | ND |
| `end` | Input | Exclusive end positions of fill intervals. Same constraints as `start`. | INT32 | ND |
| `value` | Input | Fill value for each round. Shape `(rounds,)`; dtype must match `x`. | FLOAT16, FLOAT, INT8, INT32 | ND |
| `y` | Output | Output tensor in the formula. Same shape and dtype as `x`. | FLOAT16, FLOAT, INT8, INT32 | ND |
| `axis` | Attribute | Fill axis in `[-dim(x), dim(x))`; negative values count from the last dimension. For 2-D, filling always applies to the last dimension. For 3-D, axes 0, 1, and 2 are supported. | INT | - |

## Constraints

* `value` must have the same dtype as `x`. `start` and `end` must have the same dtype, which must be INT32.
* `x` supports only 1-D, 2-D, or 3-D shapes. `start` and `end` are 2-D; `value` is 1-D. Their first dimensions (the number of fill rounds) must match.
* `rounds` must not exceed 128, and `rounds` multiplied by the effective batch count must not exceed 128. The effective batch count is `min(B, required batches)`: 1 for 1-D, the row count `N` for 2-D, and the first dimension `N` for 3-D (the `C` rows for the same `n` share one batch).
* The total number of elements in `x` must not exceed `1024 * 1024 * 1024`. When any dimension is 0 (empty tensor), the operator performs no fill and `y` is returned with the same shape as `x`.
* If the second dimension of `start` is smaller than the required row/batch count, uncovered rows or batches are not filled. Extra window columns are ignored.
* When filling along the last dimension, intervals are clamped to `[0, D)`. For 3-D with `a = 0` or `a = 1`, intervals are compared directly without clamping.

## Usage

| Interface | Sample | Description |
| :-- | :-- | :-- |
| GE graph mode | [test_geir_masked_fill_range](examples/test_geir_masked_fill_range.cpp) | Calls MaskedFillRange through the [geir graph mode](docs/MaskedFillRange_en.md). |
