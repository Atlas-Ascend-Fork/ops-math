# Contribution Guide

This project welcomes all developers to try it out and participate in contributions. Before participating in community contributions, see [cann-community](https://gitcode.com/cann/community) to learn about the code of conduct, sign the CLA, and get familiar with the contribution process of the source code repository.

## Preparation

When preparing local code and submitting PRs, developers need to pay attention to the following items:

1. When submitting a PR, fill in information such as the service background, purpose, and solution of the PR carefully according to the PR template.
2. If your change involves a new feature, a new API, a new configuration parameter, or a modified code flow (rather than a simple bug fix), discuss the solution through an issue first to prevent the code from being rejected. If you are not sure whether the change is a simple bug fix, it is recommended that you submit an issue for discussion.

## Contribution Scenarios

The main contribution scenarios for developers include the following:

- [Contributing a New Operator](#1-contributing-a-new-operator)
- [Fixing Operator Bugs](#2-fixing-operator-bugs)
- [Optimizing Operators](#3-optimizing-operators)
- [Correcting Documentation Errors](#4-correcting-documentation-errors)
- [Helping Resolve Issues Reported by Others](#5-helping-resolve-issues-reported-by-others)

---

## 1. Contributing a New Operator

The operator development contribution process is as follows:

![Operator development contribution process](./docs/en/figures/op_dev_and_contribute_flow.png "Operator development contribution flowchart")

If you want to design and implement a brand-new operator based on the NPU, you are welcome to propose your ideas and design solutions in an issue. The complete contribution process is as follows:

### 1. Create a Requirement Issue

Create a `Requirement|Requirement suggestion` issue and describe the design solution of the new operator. The issue generally needs to include the following content:

- **Background information**
- **Value/Function**
- **Design solution**

For detailed operation steps, refer to the [Issue operation guide](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md).

### 2. Requirement Review

After creating a requirement issue, you need to apply for a SIG topic and attend the Ops-basic SIG review. For the meeting schedule, topic application method, and Maintainer contact information, refer to the [Ops-basic SIG description](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/README.md).

**Handling urgent requirements:**

If a requirement is urgent, contact the Maintainer according to the Ops-basic SIG description to apply for a temporary SIG review.

**Requirement acceptance:**

If the requirement is accepted, [SIG members](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic/sig-info.yaml) will assign a proper operator category path (for example, `experimental/math`) to you. Submit the contributed operator to the corresponding operator category directory under `experimental`.

### 3. PR Submission

#### Minimum Deliverables of an Ecosystem Operator

```text
${op_class}                    # Operator category
├── ${op_name}                 # Operator name
│   ├── ${op_name}.cpp         # Operator kernel implementation file
│   ├── tests                  # Test directory
│   │   └── test_${op_name}.py # Operator test file
│   ├── CMakeLists.txt         # Operator compilation configuration file
│   └── README.md              # Operator README document
```

#### PR Submission Instructions

**Preparation before submission:**

- It is recommended that you submit a PR after the requirement review is passed.
- Before submitting a PR, complete the development environment preparation, and carefully learn about the project-specific development specifications and copyright statement requirements (for open-source code snippet citations, refer to the snippet citation guide). Ensure that your contribution complies with the project standards and that you have signed the CLA. For detailed operation steps, refer to the [PR operation guide](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md).

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverable | The operator kernel implementation and operator test files must be provided. | [fast_kernel_launch_example](examples/fast_kernel_launch_example/README_en.md) |
| Documentation deliverable | The operator README document is mandatory, and other documents can be provided as needed. | [Documentation contribution guide](docs/CONTRIBUTING_DOCS.md) |
| Precision requirement | Newly contributed operators must meet the precision standard. | [Open-source precision standard for ecosystem operators](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |

**Compliance check:**

- [ ] Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [ ] Whether the code can be compiled
- [ ] Whether the Markdown document syntax complies with the specifications

**Submission specifications:**

- **Contribution directory**: Submit the code to the specified directory `experimental/${op_class}` according to the opinions of SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Submit a PR to the target branch of the ops-math repository. Check whether the PR title is clear and whether the PR description is standardized (indicating the changed content and reason, and whether the related issue is associated).

> **NOTE**: If you want to contribute a project-standard operator, its deliverables and development process are more complex than those of an ecosystem operator, including the kernel and Tiling implementations. For the detailed contribution guide, refer to the [Appendix](#appendix).

### 4. CI Gate

Trigger the open-source repository gate by commenting with the `compile` command, and modify the code based on the CI check results. Currently, the CI gate includes the following check items:

- Code compilation
- Static check (for false positives reported by codecheck, submit them to SIG members for blocking)
- UT test
- Smoke test

After the gate is passed, @ Committer in the associated issue.

### 5. Committer Review

The Committer provides review comments after the review. Modify the code according to the comments, and @ Committer after the modification.

### 6. Maintainer Merge

After the Committer review is passed, the `/lgtm` label will be added. The Maintainer performs the final review. After confirming that there are no problems, the Maintainer adds the `/approve` label and merges the PR.

---

## 2. Fixing Operator Bugs

If you find operator bugs in this project and want to fix them, you are welcome to create an issue for feedback and tracking.

You can follow the instructions in [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Bug-Report|Bug feedback` issue to describe the bug.

---

## 3. Optimizing Operators

If you have ideas on generalization enhancement or performance optimization for some operator implementations in this project and want to implement these optimization points, you are welcome to contribute operator optimizations.

You can follow the instructions in [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Requirement|Requirement suggestion` issue to describe the optimization points and provide your design solution.

---

## 4. Correcting Documentation Errors

If you find description errors in some operator documents in this project, you are welcome to create an issue for feedback and fixes. For documentation specifications, refer to the [documentation contribution guide](docs/CONTRIBUTING_DOCS.md).

You can follow the instructions in [Submit Issue/Process Issue Task](https://gitcode.com/cann/community#提交Issue处理Issue任务) to create a `Documentation|Documentation feedback` issue to point out the problems in the corresponding document.

---

## 5. Helping Resolve Issues Reported by Others

If you have a proper solution to a problem encountered by others in the community, you are welcome to post comments in the issue to help others solve problems and pain points, and jointly improve usability.

If the issue requires code modification, enter `/assign` or `/assign @yourself` in the issue comment box to assign the issue to you, and track and help resolve the problem.

---

## Appendix

### Project-Standard Operator Deliverables

```text
${op_class}                           # Operator category
├── ${op_name}                        # Operator name
│   ├── op_host                       # Operator definition and Tiling-related implementations
│   │   ├── ${op_name}_def.cpp        # Operator definition file
│   │   ├── ${op_name}_tiling.cpp     # Operator Tiling implementation file
│   │   ├── ${op_name}_tiling_${sub_case}.cpp # Optional. Sub-scenario Tiling implementation. ${sub_case} indicates a sub-scenario (for example, arch35)
│   │   ├── ${op_name}_tiling_${sub_case}.h   # Optional. Header file of the sub-scenario Tiling implementation
│   │   └── CMakeLists.txt
│   ├── op_kernel                     # Operator kernel directory
│   │   ├── ${op_name}.cpp            # Kernel entry file, containing the main function and scheduling logic
│   │   ├── ${op_name}.h              # Kernel implementation file, defining the kernel header file, including function descriptions, structure definitions, and logic implementation
│   │   ├── ${op_name}_tiling_data.h  # TilingData file, storing configuration information related to the Tiling strategy
│   │   └── ${op_name}_tiling_key.h   # TilingKey file, defining the key of the Tiling strategy to identify different partition modes
│   ├── CMakeLists.txt                # Operator compilation configuration file, which is kept as is
│   ├── README.md                     # Operator description document
│   └── tests                         # Operator test files
│       └── ut                        # Operator UT test files
```

For the complete operator directory structure (including optional deliverables), refer to [Project Directory Structure](docs/zh/install/dir_structure.md#项目目录).

> **Description**: For Tiling implementation files that participate in compilation under the op_host directory, the file name must contain the `_tiling` identifier (for example, `${op_name}_tiling.cpp` and `${op_name}_tiling_${sub_case}.cpp`). Otherwise, the files cannot be recognized by the compilation system. When splitting Tiling implementations for sub-scenarios (for example, the specific architecture arch35), follow this naming rule.

### PR Merge Requirements

**Deliverable requirements:**

| Type | Requirement | Reference |
|------|------|---------|
| Code deliverable | The op_host operator Tiling implementation, op_kernel operator kernel implementation, and operator UT test files must be provided. | [Operator Development Guide](docs/zh/develop/aicore_develop_guide.md) |
| Documentation deliverable | The operator README document is mandatory, and other documents can be provided as needed. | [Documentation contribution guide](docs/CONTRIBUTING_DOCS.md) |

**Compliance check:**

- [ ] Whether the code complies with the [C++ Coding Standards](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)
- [ ] Whether the code complies with the basic programming specifications of standard operators
- [ ] Whether the code can be compiled
- [ ] Whether the Markdown document syntax complies with the specifications

**Submission specifications:**

- **Contribution directory**: Submit the code to the specified directory `experimental/${op_class}` according to the opinions of SIG members. You can refer to the placement rules of existing operator files.
- **PR submission**: Submit a PR to the target branch of the ops-math repository. Check whether the PR title is clear, whether the PR description is standardized (indicating the changed content and reason, and whether the related issue is associated), and whether the CLA has been signed.
