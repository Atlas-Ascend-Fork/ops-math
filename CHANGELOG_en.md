# CHANGELOG

> This document records the important changes of each version. Versions are listed in reverse chronological order.

## v8.5.0-beta.1

Release date: 2025-12-30

The first Beta version v8.5.0-beta.1 of ops-math has been released.
This version introduces multiple new features, bug fixes, and performance improvements, and is still in the testing phase.
Community feedback is sincerely welcome to further improve the stability and functional completeness of ops-math.
For usage instructions, refer to the [official documentation](https://gitcode.com/cann/ops-math/blob/master/README_en.md).

### 🔗 Version Address

[CANN 8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directory is described as follows:
├── aarch64                 # The CPU is of the ARM type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
├── x86_64                   # The CPU is of the X86 type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**Compatibility between the ops-math subpackage and related components and CANN versions**

| CANN Subpackage Version | Source Code Tag   | Compatible CANN Version|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 Key Features

- [Engineering capability] Support for math ONNX operator plugins. ([#431](https://gitcode.com/cann/ops-math/pull/431))
- [Operator implementation] New support for the concat ([#573](https://gitcode.com/cann/ops-math/pull/573)), lerp ([#519](https://gitcode.com/cann/ops-math/pull/519)), drop_out_v3 ([#539](https://gitcode.com/cann/ops-math/pull/539)), and other operators in multiple categories.
- [Documentation optimization] A quick start guide is added, and the new operator contribution process in the contribution guide is optimized. ([#472](https://gitcode.com/cann/ops-math/pull/472))
- [Usability improvement] The kernel compilation process is optimized, and info-level log printing is enabled. ([#326](https://gitcode.com/cann/ops-math/pull/326))

### 🐛 Bug Fixes

- The aclnn API of the tile operator is inconsistent with the operator prototype, causing built-in operator invocation failures. ([Issue239](https://gitcode.com/cann/ops-math/issues/239))
- An error is reported when operators in the experimental directory are executed by using the op_api that comes with the code repository. ([Issue143](https://gitcode.com/cann/ops-math/issues/143))
- The operator deployment path is inconsistent with the specified vendor_name. ([Issue86](https://gitcode.com/cann/ops-math/issues/86))
- Operator configuration files and folders cannot be automatically created. ([Issue82](https://gitcode.com/cann/ops-math/issues/82))
