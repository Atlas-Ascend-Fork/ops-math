# aclnn Return Codes

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:49:24.131Z -->

When an aclnn API is called, common API return codes are listed in [Table 1](#table1).
For exception status code values, obtain the exception information through the aclGetRecentErrMsg API ([*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi)), and then troubleshoot the problem based on the error message or contact technical support.

**Table 1**  Return Status Codes <a id="table1"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | Success. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter verification error. An illegal nullptr exists in the parameters. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter verification error. For example, the two input data types do not satisfy the input type deduction relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | An exception occurs when the API internally calls the npu runtime API. |
| ACLNN_ERR_INNER_XXX | 561xxx | An exception occurs inside the API. |

For details about more status codes of the ACLNN_ERR_INNER_XXX type, see [Table 2](#table2).

**Table 2**  Exception Status Codes <a id="table2"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER               |   561000     |    Internal exception: An internal exception occurs in the API.     |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR     |  561001      |  Internal exception: An error occurs when the API internally infers the output shape.       |
| ACLNN_ERR_INNER_TILING_ERROR     |   561002     |   Internal exception: An exception occurs when the API internally performs tiling for the npu kernel.     |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR   |   561003  |  Internal exception: An exception occurs when the API internally searches for the npu kernel (possibly because the operator binary package is not installed).    |
| ACLNN_ERR_INNER_CREATE_EXECUTOR     |    561101    |   Internal exception: The API fails to create aclOpExecutor internally (possibly due to an operating system exception).      |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR     |  561102      |   Internal exception: uniqueExecutor ReleaseTo is not called inside the API.      |
| ACLNN_ERR_INNER_NULLPTR            |   561103     |    Internal exception: An exception occurs inside the aclnn API, and a nullptr exception occurs.     |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE     |   561104     |   Internal exception: An exception occurs inside the aclnn API. The number of operator attributes is incorrect.      |
| ACLNN_ERR_INNER_KEY_CONFILICT (deprecated)     |   561105     |    **Deprecated. Use the latest ACLNN_ERR_INNER_KEY_CONFLICT.**     |
| ACLNN_ERR_INNER_KEY_CONFLICT             |   561105     |   Internal exception: An exception occurs inside the aclnn API. A hash key conflict occurs when the kernel of the operator is matched. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE     |   561106     |   Internal exception: An exception occurs inside the aclnn API. The implementation mode parameter of the operator is incorrect.    |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND     |  561107     |  Internal exception: An exception occurs inside the aclnn API. The required environment variable ASCEND_OPP_PATH is not detected.       |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED     |  561108      |   Internal exception: An exception occurs inside the aclnn API. The API fails to load the operator information JSON file in the operator kernel library.      |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND     |   561109      |   Internal exception: An exception occurs inside the aclnn API. The API fails to load a field of the operator information JSON file in the operator kernel library.      |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID     |  561110     |     Internal exception: An exception occurs inside the aclnn API. The format field in the operator information JSON file in the operator kernel library is set to an invalid value.    |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID     |    561111      |     Internal exception: An exception occurs inside the aclnn API. The dtype field in the operator information JSON file in the operator kernel library is set to an invalid value.    |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND     |   561112    |    Internal exception: An exception occurs inside the aclnn API. The binary kernel library of the operator is not loaded.     |
| ACLNN_ERR_INNER_OP_FILE_INVALID     |  561113     |   Internal exception: An exception occurs inside the aclnn API when the fields of the operator JSON file are loaded.   |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND     |  561114      |  Internal exception: An exception occurs inside the aclnn API. The number of operator attributes is inconsistent with that in the operator information JSON file and exceeds the number of attrs specified in the JSON file.       |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH     |   561115      |   Internal exception: An exception occurs inside the aclnn API. The number of operator attributes is inconsistent with that in the operator information JSON file and is less than the number of attrs specified in the JSON file.      |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE     |   561116     |   Internal exception: An exception occurs inside the aclnn API. The number of operator inputs exceeds the limit of 32.      |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL     |   561117     |  Internal exception: An exception occurs inside the aclnn API. The description in the operator information JSON file is incomplete.       |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID     |  561118     |    Internal exception: An exception occurs inside the aclnn API when the workspace information in the static binary JSON file is parsed.     |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID     |  561119      |    Internal exception: An exception occurs inside the aclnn API when the core usage information in the static binary JSON file is parsed.  |
