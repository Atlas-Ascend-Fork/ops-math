# Deterministic Computation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:53:08.727Z -->

## Introduction

During application development, due to differences in the CANN version or NPU model, the results of multiple runs of the same operator may not be exactly the same. Even under the same conditions with the same random seed, results of multiple NPU executions may differ. This difference usually arises from asynchronous multi-thread execution in the operator implementation, which changes the order of floating-point accumulation.

Deterministic computation requires the same running conditions, such as the platform, device, software version, input, and random parameters. It does not guarantee consistent results across CANN versions or NPU models.

In some special use cases, such as experiments, debugging, and regression tests, an operator is required to produce the same output across multiple runs to locate faults or analyze algorithms. This solution is generally referred to as "deterministic computation".

## Notes

- Generally, deterministic computation is not recommended. Deterministic computation of the same operator is usually slower than non-deterministic computation and may degrade the performance of a single model run. However, in use cases such as experiments, debugging, and regression tests, where consistent results across multiple runs are required to locate faults or analyze algorithms, deterministic computation improves efficiency.

- Thread note: Setting determinism multiple times in one thread is currently not recommended. For the configuration scope and validity rules, see the *Runtime Runtime API* documentation of the corresponding version.

## Usage

Currently, CANN operators are mainly called through the aclnn API or the PyTorch API (provided by the external TorchNPU framework). Some operator APIs default to a deterministic implementation, and some default to a non-deterministic implementation. For operators with a non-deterministic implementation, some support deterministic computation through manual configuration.

- **Calling the aclnn API**

  In this use case, enable deterministic computation through the aclrtSetSysParamOpt API (process-level) under "Runtime Configuration > aclrtSetSysParamOpt" in the [*Runtime Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=1` to enable deterministic computation.

- **Calling the PyTorch API**

  In this use case, enable deterministic computation using the PyTorch native switch (`torch.use_deterministic_algorithms`).

For operator APIs of different frameworks, refer to the specific aclnn API documentation or PyTorch API documentation for the default deterministic computation implementation.
