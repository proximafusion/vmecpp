# VMEC++ large tests

Suite of VMEC++ C++ tests that use large test files stored in Git LFS.

These tests live directly in the [vmecpp](https://github.com/proximafusion/vmecpp) repository
under `src/vmecpp/cpp/vmecpp_large_cpp_tests/`.

To build and run all tests:

```
cd src/vmecpp/cpp
bazel test --config=opt //vmecpp/... //vmecpp_large_cpp_tests/...
```
