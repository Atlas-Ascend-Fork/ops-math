# Documentation Center

English | [简体中文](./README.md)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-18T08:28:17.663Z pushedAt=2026-11-18T08:43:54.614Z nl=2 -->

## Directory Structure

The `docs` directory structure is described as follows:

```text
├── zh
│    ├── context                            # Common documents, such as terminology and basic concepts
│    ├── debug                              # Operator debugging guide document
│    │   ├── op_debug_prof.md
│    │   ├── ...
│    ├── develop                            # Operator development guide document
│    │   ├── aicore_develop_guide.md
│    │   ├── aicpu_develop_guide.md
│    │   ├── ...
│    ├── figures                            # Image directory
│    ├── install                            # Environment installation and compilation guide document
│    │   ├── build.md
│    │   ├── compile.md
│    │   ├── quick_install.md
│    │   └── ...
│    ├── invocation                         # Operator call guide document (including aclnn call, graph mode call, etc.)
│    │   ├── quick_op_invocation.md
│    │   ├── ...
│    ├── menu_aclnn_api.md                  # Full aclnn API index file
│    ├── op_api_list.md                     # aclnn API list
│    ├── op_list.md                         # Full operator list
├── en                                     # English documents, structure mirrors zh
│    ├── context                            # Common documents, such as terminology and basic concepts
│    ├── debug                              # Operator debugging guide document
│    ├── develop                            # Operator development guide document
│    ├── figures                            # Image directory
│    ├── install                            # Environment installation and compilation guide document
│    ├── invocation                         # Operator call guide document
│    ├── menu_aclnn_api.md                  # Full aclnn API index file
│    ├── op_api_list.md                     # aclnn API list
│    ├── op_list.md                         # Full operator list
├── CONTRIBUTING_DOCS.md                 # Document contribution description
├── QUICKSTART.md                        # Quick start
└── README.md
```

## Advanced Tutorials

> Chinese readers can go to the [中文文档中心](./README.md) directly; it mirrors this page section by section.

### Guide Documentation

| Documentation                                                         | Description                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Build from Source Code](en/install/compile.md)                        | Introduces different source code build methods and verification methods in online/offline scenarios.          |
| [Operator Call](en/invocation/quick_op_invocation.md)         | Introduces the method of calling operator examples and different operator call methods (such as PyTorch/aclnn/graph). |
| [Standard Operator Development](en/develop/aicore_develop_guide.md)       | Introduces how to define operator prototypes and implement Tiling and Kernel based on the standard project. Such operators are called "standard operators".<br>Standard operators support aclnn and graph mode calls. |
| [Simple Operator Development](../examples/fast_kernel_launch_example/README.md) | Describes how to implement fast_kernel_launch based on a simple project, that is, the `<<<>>>` method. Such operators are called "simple operators" (also known as ecosystem minimal operators).<br>Simple operators support only PyTorch calls. |
| [Operator Debugging and Tuning](en/debug/op_debug_prof.md)                    | Introduces common operator function debugging and performance tuning methods (such as data collection and simulation pipeline). |

### API Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Operator List](en/op_list.md)                        | Describes the list of all operators included in the project.                                 |
| [aclnn List](en/op_api_list.md)                   | Describes the list of all operator aclnn APIs included in the project. To facilitate operator calls on the Host side, C language APIs with the aclnn prefix are provided. |

### Tool Documentation

| Documentation        | Description                  |
| ----------------------- | ---------------------- |
| [Simulator Simulation Tool](en/debug/npu_sim.md) | A SoC-level simulation tool for operator development scenarios, used to analyze the precision and performance data of AI tasks running on the AI simulator at each stage. |

### More Documentation

Welcome to the [wiki center](https://gitcode.com/cann/ops-math/wiki/Home.md) to learn more project information, including project positioning, supplementary knowledge of operator development, performance optimization methodology, FAQ, and problem locating methods.

## Appendix

| Documentation                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [Operator Basic Concepts](en/context/basic_concept.md) | Introduces basic concepts and terminology in the operator domain, such as quantization/sparsity, data types, and data formats. |
| [build Parameter Description](en/install/build.md)   | Introduces the functions and values of the build.sh parameters in this project, including source code compilation, operator call, and debugging. |
