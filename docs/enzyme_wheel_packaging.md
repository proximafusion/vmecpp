# Shipping Enzyme-differentiated kernels in pip wheels

Status: phase 1 implemented (experimental, not yet on the release path).
Follow-up to #710 (and the #584 -> #703 -> #707 -> #710 stack), which wires
an exact Enzyme-derived JVP/VJP of the force kernel into `vmecpp_core`
behind `-DVMECPP_ENABLE_ENZYME=ON`. That flag is `OFF` by default and was,
until this change, only exercised by the dedicated
`.github/workflows/test_enzyme.yaml` job, which builds its own Clang/LLVM
21 + `ClangEnzyme-21` (pinned at `v0.0.264`) toolchain from scratch on an
`ubuntu-22.04` runner. None of that reached a wheel: `cibuildwheel`
(`[tool.cibuildwheel]` in `pyproject.toml`) still builds `vmecpp_core` with
the default compiler on each platform (`manylinux`'s `gcc`, `AppleClang` on
macOS) and never sets `VMECPP_ENABLE_ENZYME`.

`.github/workflows/build_enzyme_wheels.yaml` now implements the mechanism
described below for `linux-x86_64`, `macos-arm64` (`macos-14`) and
`macos-x86_64` (`macos-15-intel`), producing installable, smoke-tested
Enzyme-enabled wheels as workflow artifacts. It is a separate,
`workflow_dispatch`/path-filtered workflow, deliberately not wired into
`.github/workflows/pypi_publish.yml`'s `release: published` trigger yet:
none of this has been exercised against a real manylinux container or a
real macOS runner outside of this change, so it should not be trusted with
the actual release pipeline until it has run green a few times and its
wheels have been checked for the no-regression property described below.
Promoting it is a follow-up once that evidence exists. Remaining scope gaps
(`linux-aarch64`, Windows, merging into `pypi_publish.yml`) are tracked in
[Open questions](#open-questions).

## Constraints

- **No regression for non-adjoint users.** Everyone who does not call the
  exact-JVP/VJP API pays for this feature exactly nothing: same compiler,
  same flags, same object code for every other translation unit.
- **No new runtime dependency.** Enzyme is an LLVM optimization pass that
  runs once, at compile time, over the two translation units
  `exact_force_jvp.cc` / `exact_force_vjp.cc`. Its output is a plain `.o`
  with ordinary machine code; `ClangEnzyme-21.so` and `clang` itself are
  build-time tools, not something the wheel loads at import time or ships.
- **manylinux/macOS toolchain policy.** `cibuildwheel`'s manylinux
  containers build with the container's GCC. On macOS, `vmecpp_core` is
  actually built with Apple's own clang (`AppleClang`; `before-build = "brew
  install gcc libomp"` in `pyproject.toml` only supplies `gfortran` and
  OpenMP, not the C/C++ compiler) - and Apple's clang cannot load an
  upstream `ClangEnzyme` plugin (different fork, no matching `-fplugin`
  support), so prebuilt objects are not just the lower-risk option on
  macOS, they are the only one that works there at all. Installing a full
  matching upstream Clang/LLVM + built-from-source Enzyme plugin inside
  every wheel-build container, for every architecture, would also be the
  kind of toolchain surgery that risks breaking `manylinux`
  glibc-symbol-version guarantees.

## Implemented architecture: prebuilt-object mode, not a Clang wheel toolchain

`CMakeLists.txt` already has the mechanism this needs:
`VMECPP_ENZYME_JVP_OBJECT` / `VMECPP_ENZYME_VJP_OBJECT` let the two Enzyme
translation units be supplied as **prebuilt `.o` files**, compiled
elsewhere with a matching Clang + `ClangEnzyme`, and linked into
`vmecpp_core` (and `vmecpp_core_v3`) without the main build ever invoking
Clang or loading the plugin. This is the same "prebuilt Enzyme mode" the
CMake comments call out for `icx`/`icpx`; `build_enzyme_wheels.yaml` reuses
it for wheels instead of inventing a second path:

1. **A small, separate CI job** (`build-enzyme-objects` in
   `build_enzyme_wheels.yaml`) reuses the `test_enzyme.yaml` toolchain
   recipe (pinned `LLVM_VERSION`, pinned `ENZYME_REF`) on Linux, and an
   analogous Homebrew-LLVM recipe on macOS, to build a plugin-mode
   `vmecpp_core` (`-DVMECPP_ENABLE_ENZYME=ON -DVMECPP_ENZYME_PLUGIN=...`)
   and harvest the two resulting `exact_force_jvp.cc.o` /
   `exact_force_vjp.cc.o` objects straight out of the CMake build tree.
   There is only one variant per arch: the CMake rules already force
   `-march=x86-64` for these two files regardless of
   `VMECPP_HWCAPS_DISPATCH`, since Enzyme does not support the AVX mask
   intrinsics used for `x86-64-v3`. Both the plugin build and the object
   build are cached (`actions/cache`), matching how `test_enzyme.yaml`
   already caches the `ClangEnzyme` plugin build.
   - **Not yet done:** compiling inside the actual `manylinux_2_28`
     container `cibuildwheel` uses for the Linux target, so the resulting
     `.o`'s glibc-symbol-version floor is only as good as the plain
     `ubuntu-24.04` runner's toolchain, not the container's. `auditwheel`
     (run automatically by `cibuildwheel`'s manylinux repair step) does
     check this and will fail loudly, rather than silently shipping a
     wheel with an inflated glibc requirement, but this has not been
     exercised yet since the workflow does not run on every push. See
     [Open questions](#open-questions).
2. **The objects feed the normal wheel build** (`build-enzyme-wheels` job)
   via `CIBW_ENVIRONMENT_LINUX` / `CIBW_ENVIRONMENT_MACOS`, setting
   `CMAKE_ARGS="-DVMECPP_ENABLE_ENZYME=ON -DVMECPP_ENZYME_JVP_OBJECT=... -DVMECPP_ENZYME_VJP_OBJECT=..."`
   with paths translated for the Linux case, where `cibuildwheel` mounts
   the repo at `/project` inside the manylinux container. No Clang, LLVM,
   or `ClangEnzyme` is installed inside the `cibuildwheel` container or
   macOS build environment itself; only the two small `.o` files, produced
   by the sibling job, are downloaded first.
   - `vmecpp_core` and `vmecpp_core_v3` continue to be compiled with
     today's compiler and flags for every translation unit except those
     two; nothing about the default, non-adjoint code path changes.
3. **Fail closed, not silently, when a pair is missing** for a given
   target (currently true for `linux-aarch64`, not yet built): the
   consuming step requires both files to exist before setting
   `VMECPP_ENABLE_ENZYME=ON`, and the existing runtime guard (the
   "non-Enzyme build raises instead of using finite differences" behavior
   from #710) tells users why the exact-VJP API is unavailable on a wheel
   that was built without it.

This keeps the wheel-build containers untouched for the code path every
non-adjoint user exercises, and confines all the toolchain risk to one
small, cacheable, independently-testable CI job, currently isolated in
`build_enzyme_wheels.yaml` rather than the release-triggered
`pypi_publish.yml`.

## Verifying no performance regression

Because the only two translation units touched are `exact_force_jvp.cc`
and `exact_force_vjp.cc` - which are otherwise not referenced by the
production forward solve - object-identity is enough for everything else,
but should still be checked explicitly. **Implemented so far:**
`build-enzyme-wheels` installs the Enzyme-enabled wheel into a clean venv
and runs `tests/test_hessian.py` and `tests/test_external_optimizers.py`
against it, i.e. functional correctness of the shipped wheel, not yet
performance. **Still to do:**
- Add a CI check that builds `vmecpp_core` twice, once with
  `VMECPP_ENABLE_ENZYME=OFF` and once in prebuilt-object mode, and diffs
  the resulting `.so` symbol table / disassembly for every translation
  unit *other* than the two Enzyme ones (e.g. `objdump -d` per `.o`
  before linking, or comparing `nm --defined-only` output for unrelated
  symbols) to catch any accidental change in compiler flags reaching the
  rest of the library.
- Extend the existing `benchmarks/benchmark_exact_hvp.py` run (already
  in `test_enzyme.yaml`) to also record a plain forward-solve timing
  (e.g. the W7-X or CTH example already used elsewhere in the test suite)
  under both configurations, and gate on the two forward-solve timings
  matching within noise. This is what actually answers "does an
  Enzyme-enabled wheel regress non-adjoint users", as opposed to the
  HVP-vs-finite-difference comparison the current benchmark performs.
- Run this comparison against the actual wheels `build_enzyme_wheels.yaml`
  produces (not just the local CMake build used by `test_enzyme.yaml`),
  since the wheel is the artifact users install.

## Open questions

- Which wheel targets get Enzyme objects at all. `linux-aarch64` and
  `musllinux` are not yet built by `build_enzyme_wheels.yaml`; they add
  real CI matrix cost for the object-build job (and, for `linux-aarch64`,
  its own native-arm64 Clang/Enzyme toolchain build). Extending to them is
  the natural next step once the current three targets have run green.
- Whether the Linux objects should instead be compiled inside the actual
  `manylinux_2_28` container (e.g. via a `docker run` step using that
  image, with LLVM/Clang from `conda-forge` rather than `apt.llvm.org`
  since the container has no APT), so the glibc-symbol-version floor
  matches the wheel exactly instead of relying on `auditwheel` to catch a
  mismatch after the fact. Currently `build_enzyme_wheels.yaml` builds the
  objects on the plain `ubuntu-24.04` runner.
- Whether prebuilt objects should be committed as versioned release
  assets (rebuilt on a schedule/on relevant source changes) or built
  fresh in every wheel release; the former is cheaper per release but
  needs its own staleness check against `exact_force_jvp.cc` /
  `exact_force_vjp.cc` content hashes and the pinned `ENZYME_REF`.
  `build_enzyme_wheels.yaml` currently rebuilds on every run (relying on
  `actions/cache` for the plugin build only).
- When to merge `build_enzyme_wheels.yaml`'s environment variables into
  `pypi_publish.yml`'s cibuildwheel step so real releases ship the Enzyme
  kernels. Proposed gate: several green runs of this workflow, plus the
  performance-regression checks above passing on the produced wheels.
- Windows is currently outside `test_enzyme.yaml`'s scope entirely (no
  pinned Clang/Enzyme recipe exists for it yet); this stays out of scope
  here too, only packaging what already exists for Linux/macOS.
