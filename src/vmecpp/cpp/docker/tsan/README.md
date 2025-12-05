# OpenMP-aware ThreadSanitizer

This is a tutorial how to build `clang` and `libomp` with ThreadSanitizer (TSan)
support for being able to check OpenMP-parallelized C/C++ code.

## Perform ThreadSanitizer runs on code in the Proxima Fusion monorepo

1. Build the docker container:

   ```bash
   cd docker/tsan
   docker build -t proximafusion/tsan .
   ```

   Use `--no-cache` to discard a previous build step.

   Use `--progress=plain` to circumvent the console overwriting stuff in place.

2. Run the docker container from within `docker/tsan`:

   ```bash
   docker run --name tsan --rm -it -v ~/code/vmecpp:/vmecpp proximafusion/tsan /bin/bash
   ```

   Note that the directory at which this repository is cloned is assumed to be
   `~/code/vmecpp` here. Adjust this for the folder where you cloned the repo.

3. Inside the docker container, navigate to the bind-mounted C++ source code folder:

   ```bash
   cd /vmecpp/src/vmecpp/cpp
   ```

4. Clear the host's Bazel cache that was exposed to the docker container via the
   bind-mount:

   ```bash
   bazel clean --expunge
   ```

5. Make sure the `LD_LIBRARY_PATH` environment variable is set to the location of
   the `libarcher.so` library in the docker container:

   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib/x86_64-unknown-linux-gnu
   ```

   TODO(jons): fix this - should not be needed (?)

6. Run the unit tests for VMEC++ under TSan/Archer:

   ```bash
   bazel test --config=tsan_archer_in_docker --test_output=streamed --nocache_test_results -- //vmecpp/... -//vmecpp/vmec/pybind11/...
   ```

   `pybind11` is disabled for now, as it does not compile yet with the new
   `clang`.

7. Run example cases using the TSan-enabled executable:

   ```bash
   export ARCHER_OPTIONS="verbose=1"
   export TSAN_OPTIONS="ignore_noninstrumented_modules=1"
   export OMP_NUM_THREADS=2
   bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone vmecpp/example_inputs/solovev.json
   bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone vmecpp/example_inputs/cth_like_fixed_bdy.json
   bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone vmecpp/example_inputs/cma.json
   bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone vmecpp/example_inputs/test.vmec.json
   bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone vmecpp/example_inputs/w7x_ref_167_12_12.json
   ```

8. Clean up the Bazel cache after the TSan/Archer work is done:

   ```bash
   docker> exit
   host> bazel clean --expunge
   ```

When running the tests, make sure that the following message appears:

`Archer detected OpenMP application with TSan, supplying OpenMP synchronization semantics`

This is enabled by the environment variable `ARCHER_OPTIONS="verbose=1"` and, in
particular the second part of that message, indicates that the TSan/Archer setup
for OpenMP-aware thread sanitization works.

Some background info on this setup is available in `background_info.md`.
