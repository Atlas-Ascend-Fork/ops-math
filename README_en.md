<div align="center">

# ops-math

English | [简体中文](./README.md)

[![CANNcommunity](https://img.shields.io/badge/docs-社区全量算子-brightgreen.svg?style=flat)](https://www.hiascend.com/document/redirect/CannCommunityOplist)
[![repo](https://img.shields.io/badge/docs-项目文档-green.svg?style=flat)](docs)
[![license](https://img.shields.io/badge/license-CANN_Open_2.0-blue.svg)](LICENSE)
[![contributing](https://img.shields.io/badge/贡献指南-teal)](CONTRIBUTING_en.md)
[![SIG](https://img.shields.io/badge/SIG-ops--basic-yellow)](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/hicann/ops-math)

</div>

---

## 🔥Latest News

- [2026/07] Added support for [operators and new features](https://gitcode.com/cann/ops-math/wiki/Ascend950重点特性与本仓算子样例介绍.md) of the Huawei Ascend 950 series, covering operators such as CumulativeLogSumExp, MatrixSetDiagV2, DropoutV3Grad, SignBitsUnpack, and ProdVirialSeA; introduced new features such as FLOAT8/FLOAT4 low-precision type support, Philox PRNG random number generation, and Reg API migration.
- [2026/01] Added [QuickStart](docs/QUICKSTART.md), which guides beginners to get started with operator project deployment from scratch (Docker environment supported) and with the operator development and contribution workflows.
- [2025/12] Open-source operators support the Ascend 950PR&950DT products, and you can develop and debug them with the CANN Simulator simulation tool. A <<<>>> kernel heterogeneous invocation example is added to the add operator, making it convenient for you to customize and use it. Newly supported operators in multiple categories include [concat](conversion/concat/), [lerp](math/lerp/), and [drop_out_v3](random/drop_out_v3/).
- [2025/11] Improved the README descriptions of multiple operators, and improved the operator development sample documents and examples.
- [2025/10] Added the experimental directory, improved the [contribution guide](CONTRIBUTING_en.md), and enabled developers to debug and contribute custom operators.
- [2025/09] The ops-math project went online for the first time.

## 🚀Overview

ops-math is a basic operator library that provides numerical computation in the [CANN](https://hiascend.com/software/cann) (Compute Architecture for Neural Networks) operator library. It includes conversion-class, math-class, and random-class operators, covering scenarios such as tensor shape transformation, basic mathematical operations, and random number generation. The position of the sub-library in the architecture diagram is as follows.

<img src="docs/en/figures/architecture.png" alt="Architecture diagram"  width="700px">

## 📌Version Compatibility

The source code of this project is released together with CANN software versions. For the mapping between CANN software versions and the tags of this project, refer to the corresponding version notes in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure smooth custom development based on the source code, select a compatible CANN version and GitCode tag source code. Using the master branch may cause version mismatch risks.

## 🛠️Environment Preparation

[Environment deployment](docs/zh/install/quick_install.md) is a prerequisite for experiencing the capabilities of this project. Install the NPU driver and CANN package first, and ensure the environment works properly.

## ⬇️Source Code Download

After the environment is ready, download the branch source code that matches the CANN version. The general command is as follows, where \$\{tag\_version\} is replaced with the branch tag name. The following uses the download of the 9.0.0 branch source code as an example:

```bash
# General command: git clone -b ${tag_version} https://gitcode.com/cann/ops-math.git
git clone -b 9.0.0 https://gitcode.com/cann/ops-math.git
```

> Note: If the matching branch source code already exists in the environment, **you can skip this step**. For example, CANNLab provides the source code corresponding to the latest CANN version by default.

## 📖Learning Tutorials

- [Quick start](docs/QUICKSTART.md): Quickly experience the core basic capabilities of the project from scratch, covering source code compilation, operator invocation, development, debugging, and other operations.
- [Advanced tutorials](docs/README.md): If you need to deeply understand project compilation and deployment, operator invocation, development, debugging and tuning, and other capabilities, go to the documentation center for detailed guidance.

## 💬Related Information

- [Directory structure](docs/zh/install/dir_structure.md)
- [Contribution guide](CONTRIBUTING_en.md)
- [Security statement](SECURITY_en.md)
- [License](LICENSE)
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)
- [Committer list](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md#committer%E5%88%97%E8%A1%A8)

---
PS: The features and documentation of this project are continuously updated and improved. Stay tuned for the latest version.

- **Issue feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/ops-math/issues).
- **Community interaction**: Join discussions through GitCode [Discussions](https://gitcode.com/cann/ops-math/discussions).
- **Technical column**: Read technical articles through GitCode [Wiki](https://gitcode.com/cann/ops-math/wiki).
