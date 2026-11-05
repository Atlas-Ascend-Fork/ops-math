# AdjacentDifference op_api UT Coverage Report

## Conclusion

**op_api/adjacent_difference.cpp line coverage: 0% → 100% (7/7 lines), function coverage 100% (1/1)**

## Root Cause

The original test file `tests/ut/op_api/test_adjacent_difference_l0.cpp` does not conform to the CMake collection rule:
in `cmake/ut.cmake`, `file(GLOB OPAPI_CASES_SRC ${MODULE_DIR}/test_aclnn_*.cpp)` collects only
case files with the `test_aclnn_` prefix. Therefore, the original cases were never compiled into the `math_op_api_ut` executable,
no gcda file was generated for the function `l0op::AdjacentDifference`, and the coverage was 0.

## Fix

Added `tests/ut/op_api/test_aclnn_adjacent_difference.cpp` (conforming to the glob rule), with 19 cases in total:

| Case Category | Count | Description |
|---------|------|------|
| Full coverage of input dtypes × output INT32 | 11 | FLOAT/FLOAT16/BF16/INT8/INT16/INT32/INT64/UINT8/UINT16/UINT32/UINT64 |
| Representative input dtypes × output INT64 | 3 | FLOAT/INT32/INT64 |
| Shape boundaries | 5 | 1D/5D/empty tensor {0,3}/single element {1}/non-contiguous stride |

Each case verifies: the output is not empty, the output shape is the same as the input shape (GetViewShape), and the output dtype is the specified yDtype (GetDataType).

## Verification Environment

- CANN: 9.2.0-beta.1 (Ascend 950PR, arch35)
- Build: `bash build.sh -u --opapi --ops=adjacent_difference --soc=ascend950 --cov --noexec`
- Execution: `./build/tests/ut/op_api/math_op_api_ut --gtest_filter=*AdjacentDifference*`
  → 19 tests, 19 passed, 0 failed
- Coverage collection: lcov 2.0 (coverage.info is a local intermediate artifact and is not committed to the repository; the raw data has been excerpted into the line-by-line coverage details below)

## Line-by-Line Coverage Details (adjacent_difference.cpp)

| Line | Hit Count | Content |
|------|---------|------|
| 29 | 19 | Function entry `l0op::AdjacentDifference` |
| 31 | 19 | OP_LOGI |
| 32 | 19 | L0_DFX |
| 34 | 19 | AllocTensor(x->GetViewShape(), yDtype) |
| 36 | 19 | ADD_TO_LAUNCHER_LIST_AICORE (stub mode) |
| 37 | 19 | OP_LOGI Launch finish |
| 38 | 19 | return y |
