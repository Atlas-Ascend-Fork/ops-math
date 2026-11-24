# ViewCopy 头文件安装回归测试

本测试调用仓库真实的 `add_modules_sources`、`gen_aclnn_with_opdef` 和安装规则，收集常规 Contiguous 模块头文件。随后仅从暂存安装目录读取 ViewCopy 头文件及汇总头文件，核对两段式函数签名，并链接 CANN 库验证符号可用。无需 NPU，不执行算子。

```bash
cmake -S tests/cmake/view_copy_header -B /tmp/view-copy-header \
  -DCANN_ROOT=/usr/local/Ascend/cann-9.3.0/x86_64-linux \
  -DCMAKE_INSTALL_PREFIX=/tmp/view-copy-header/install
cmake --install /tmp/view-copy-header
cmake --build /tmp/view-copy-header --target consumer
/tmp/view-copy-header/consumer
```

自定义包路径验证使用独立构建目录，并增加 `-DENABLE_CUSTOM=ON`。常规安装应包含 `include/aclnnop/aclnn_view_copy.h` 和 `include/aclnnop/level2/aclnn_view_copy.h`；自定义安装应包含 `vendors/header_test_math/op_api/include/aclnn_view_copy.h`。安装件中的旧版本不含该头文件，因此编译调用方应失败。
