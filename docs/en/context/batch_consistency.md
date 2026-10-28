# Batch Consistency

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:50:11.992Z -->

## Introduction

During application development, to achieve higher performance, partial operators may produce different computation results for the same Token in different batch sizes or at different positions in the same batch.
Currently, when the running environment is the same, you can configure the computation process of partial operators to use the Batch consistency algorithm, ensuring that the computation results remain identical regardless of how the inputs are combined.
**Batch consistency algorithm**: For a given Token, the output must be bitwise identical, regardless of the position of the Token in the batch, the batch size, or the other tokens batched with it.

## Notes

- Usually, do not enable Batch consistency computation, because enabling it for an operator causes certain performance degradation in related operators, and the single-run performance of the model may decrease. However, in use cases such as experiments, debugging, and regression tests, where results of multiple runs must be identical to locate problems and test algorithms, Batch consistency computation improves efficiency.

- Currently, the configuration is a process-level switch.

- **Version constraints**: TorchNPU 26.2.0 or later and CANN 9.2.0 or later.

## Usage

Currently, CANN operators are mainly called through the aclnn API or PyTorch API (torch_extension). Partial operator APIs default to a Batch consistency implementation, while others default to a non-Batch consistency implementation. For operators with a non-Batch consistency implementation, some can be manually configured to use Batch consistency.

- **Calling the aclnn API**

  In this use case, enable Batch consistency through the "Runtime Configuration > aclrtSetSysParamOpt" API (process-level) in [*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=3` to enable Batch consistency computation.

- **Calling the PyTorch API**

  In this use case, enable Batch consistency computation through the `torch_npu.npu.set_deterministic_level` API in [*TorchNPU Custom APIs*](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/overview.md).

For operator APIs of different frameworks, refer to the specific aclnn API documentation or PyTorch API documentation for the default implementation strategy of Batch consistency computation.
