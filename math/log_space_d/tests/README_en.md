# LogSpaceD Test Deliverable List

English | [简体中文](./README.md)

Only the formal Golden, repeatable ST cases, and Host UT source code are delivered. Generation and audit scripts, as well as test reports, logs, and compilation caches, are not part of the deliverables of the target repository.

| Location (ST files are all in st/arch35) | Purpose | Count |
| --- | --- | ---: |
| assets/golden.py | The only formal TTK Golden. The CPU Promote allows an FP64 container and does not change the production dtype contract | — |
| ttk_kernel_log_space_d_st.csv | Basic positive cases | 255 |
| ttk_kernel_log_space_d_coverage_st.csv | Supplements for values, default attributes, carriers, core splitting, and large index boundaries | 330 |
| ttk_kernel_log_space_d_non_arange_st.csv | Independent smoke case with assist=[37] | 1 |
| ttk_kernel_log_space_d_coverage_negative.csv | Kernel invalid parameters and input/output descriptors. The scalar additionally requires GEIR verification | 34 |
| exception_geir.csv | GEIR/Kernel exception probes and valid control groups | 21 |
| exception_attribute_transport.csv | Finiteness and transport boundaries of GEIR Float attributes | 7 |
| ttk_geir_log_space_d_infer_coverage.csv | Explicit [-1]/[-2] graph building descriptions, while the runtime input is still [1] | 24 |
| ttk_geir_log_space_d_determinism.csv | Repeated execution checks of the existing positive subset, which are not counted in the number of new positive cases | 27 |
| ut/op_host/test_log_space_d_infershape.cpp | Formal inference rule UT | — |
| ut/op_host/arch35/test_log_space_d_tiling_arch35.cpp | Tiling contract and boundary UT | — |

The three positive CSVs contain 586 unique cases in total, which can be used for Kernel or GEIR. Performance testing uses the same full list, and no duplicate performance subset is delivered. The input is fixed to the one-dimensional [1], and the output is the one-dimensional [steps].

Positive tests use native TTK: `--plugin <算子目录>/tests/assets/golden.py --compare cross_check --golden-mode Promote --pc 1 --xpu-perf -c true -d true`. Kernel performance uses `--task-prof true --run 100`; the GEIR determinism special test uses `--deterministic-level 1 --run 10`.

Exception tests use `--golden-mode Disable --task-prof false`, which respectively explain valid controls, target code rejections, and framework early rejections. The native PASS/FAIL totals cannot replace the interception conclusions. The Kernel normalizes a scalar to [1]. The GEIR bare Infinity transport also has limitations, and the corresponding limitations cannot be used as evidence that the operator passes.

The delivered source code retains only the formal V2 InferShape and does not include the local temporary V1 bridge. Rebuild before execution, and confirm through smoke testing that the runtime environment is actually dispatched to the target V2 and Kernel. Historical V1 bridge test results cannot replace the V2-only path verification. The cases are not bound to V1 callbacks, but the executability of GEIR still depends on the registration dispatch support of the CANN environment.

The case generation and contract audit scripts are located in the generation project `LogSpaceD_package/tests/case_generation`. All execution results are directed through the TTK `-o` option to a new directory inside the generation project. Overwriting the frozen final results or writing to the target repository is forbidden.
