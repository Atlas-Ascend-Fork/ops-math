# ViewCopy header installation regression test

This test uses the repository's actual `add_modules_sources`, `gen_aclnn_with_opdef`, and installation rules to collect headers from the regular Contiguous module. It then reads the ViewCopy and aggregate headers exclusively from the staged installation directory, checks the two-phase function signatures, and links against the CANN library to verify that the symbols are available. No NPU is required, and the operator is not executed.

```bash
cmake -S tests/cmake/view_copy_header -B /tmp/view-copy-header \
  -DCANN_ROOT=/usr/local/Ascend/cann-9.3.0/x86_64-linux \
  -DCMAKE_INSTALL_PREFIX=/tmp/view-copy-header/install
cmake --install /tmp/view-copy-header
cmake --build /tmp/view-copy-header --target consumer
/tmp/view-copy-header/consumer
```

To verify the custom package path, use a separate build directory and add `-DENABLE_CUSTOM=ON`. The regular installation should contain `include/aclnnop/aclnn_view_copy.h` and `include/aclnnop/level2/aclnn_view_copy.h`; the custom installation should contain `vendors/header_test_math/op_api/include/aclnn_view_copy.h`. The previously installed version does not contain this header, so compiling a consumer against that version should fail.
