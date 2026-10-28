# Two-Phase API

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:54:55.257Z -->

When calling an operator API based on the single-operator API execution mode, the process typically follows a "two-phase" pattern, as shown below:

```cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-phase API aclxxXxxGetWorkspaceSize to calculate the amount of workspace memory required for this API call. After obtaining the required workspaceSize, allocate NPU memory based on workspaceSize, and then call the second-phase API aclxxXxx to perform the computation.

Here, "aclxx" represents the operator interface prefix, such as aclnn, and "Xxx" represents the corresponding operator type, such as the Add operator.

> NOTE:
>
> - workspace refers to the temporary memory, other than inputs/outputs, required by an operator to complete computation on the NPU. workspaceSize indicates the size of the temporary memory.
> - The second-phase API aclxxXxx(...) cannot be called repeatedly. The following call pattern causes an exception:
>
>   ```cpp
>   aclxxXxxGetWorkspaceSize(...)
>   aclxxXxx(...)
>   aclxxXxx(...)
