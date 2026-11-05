# AddExamplePypto

English | [简体中文](./README.md)

This is the PyPTO version of `examples/add_example`. The function is also element-wise addition. The difference is that the kernel is written in the PyPTO DSL (`op_kernel/add_example_pypto.py`) instead of AscendC C++. This operator is used to demonstrate how PyPTO operators are integrated in ops-cv.

## Compilation

```bash
source <ascend-toolkit>/set_env.sh
bash build.sh --pkg --ops=add_example_pypto --soc=ascend950
```

## Integration Method

In the `op_host/CMakeLists.txt` file of the operator, add two lines **before** `add_modules_sources` (or `add_all_modules_sources`):

```cmake
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

add_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
```

- `require_pypto_pro(<op_name>)`: If `pypto_pro` does not exist in the environment, a warning is reported and this operator is skipped. The entire configure does not fail.
- `enable_pypto_kernel(<op_file>)`: `<op_file>` must be consistent with `op_kernel/<op_file>.py` and `ExtendCfgInfo("opFile.value", ...)` in the operator definition.

## Build Process

`enable_pypto_kernel` calls `scripts/util/pypto_codegen.py` in the **configure phase** to perform codegen for the `.py` file of the kernel:

| Artifact | Purpose |
| --- | --- |
| `<Tiling>_tiling.h` | Tiling struct, which is directly used by the host-side tiling |
| `<TilingKey>_tilingkey.h` | `ASCENDC_TPL` tiling key declaration |
| `<op_file>_pypto_infer.cpp` | Self-contained source code for `.i` infer |

The preceding two header files are **force-included** in the `*_tiling*.cpp` files of this operator in the `-include` mode (in the order: tilingkey first, then tiling).
Therefore, the host-side tiling can directly use the tiling struct defined in the `.py` file without an explicit include:

```cpp
// AddExamplePyptoTiling comes from op_kernel/add_example_pypto.py and is visible through force-include
AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
```

Therefore, the host and kernel share the same tiling layout and tiling key bit allocation, and no manually written copy is required.

In the **build phase**, the wrapper generated for this operator (`<build>/tbe/dynamic/<op_file>.py`) calls `pypto_compile_op`
instead of `compile_op`. It performs codegen based on the tiling key and reuses the AscendC backend to compile the kernel binary.

## Chip and Arch

The codegen arch of PyPTO is automatically derived by the platform (`a2`/`a3`/`a5`). This operator registers only `ascend950` (a5), which is consistent with the arch automatically derived by the current toolkit.
When compiling the kernel binary, cmake exports `PYPTO_JIT_ARCH` based on the soc being compiled
(`ascend910b`/`ascend910_93` → `a3`, `ascend950` → `a5`), preventing `pypto_compile_op` from falling back to the default `a5`
and generating instructions not supported by the target chip.
