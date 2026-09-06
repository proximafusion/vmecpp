# Shipping Enzyme-differentiated kernels in pip wheels

Status: proposal. Follow-up to #710 (and the #584 -> #703 -> #707 -> #710
stack), which wires an exact Enzyme-derived JVP/VJP of the force kernel into
`vmecpp_core` behind `-DVMECPP_ENABLE_ENZYME=ON`. That flag is `OFF` by
default and is currently only exercised by the dedicated
`.github/workflows/test_enzyme.yaml` job, which builds its own Clang/LLVM
21 + `ClangEnzyme-21` (pinned at `v0.0.264`) toolchain from scratch on an
`ubuntu-22.04` runner. None of that reaches a wheel: `cibuildwheel`
(`[tool.cibuildwheel]` in `pyproject.toml`) still builds `vmecpp_core` with
the default compiler on each platform (`manylinux`'s `gcc`, Homebrew `gcc`
on macOS) and never sets `VMECPP_ENABLE_ENZYME`. This note proposes closing
that gap.

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
  containers and Homebrew's default macOS toolchain are GCC-based (or an
  Apple-clang ABI on macOS that is untested against upstream
  `ClangEnzyme`). Installing a full matching upstream Clang/LLVM +
  built-from-source Enzyme plugin inside every wheel-build container,
  for every architecture, is exactly the kind of toolchain surgery that
  breaks `manylinux` glibc-symbol-version guarantees and roughly doubles
  the wheel build matrix's fragility.

## Proposed architecture: prebuilt-object mode, not a Clang wheel toolchain

`CMakeLists.txt` already has the mechanism this needs:
`VMECPP_ENZYME_JVP_OBJECT` / `VMECPP_ENZYME_VJP_OBJECT` let the two Enzyme
translation units be supplied as **prebuilt `.o` files**, compiled
elsewhere with a matching Clang + `ClangEnzyme`, and linked into
`vmecpp_core` (and `vmecpp_core_v3`) without the main build ever invoking
Clang or loading the plugin. This is the same "prebuilt Enzyme mode" the
CMake comments call out for `icx`/`icpx`. Reuse it for wheels instead of
inventing a second path:

1. **Add a small, separate CI job** (`build_enzyme_objects.yaml`) that
   reuses the `test_enzyme.yaml` toolchain recipe (pinned `LLVM_VERSION`,
   pinned `ENZYME_REF`) to compile `exact_force_jvp.cc` and
   `exact_force_vjp.cc` for each wheel target
   (`linux-x86_64`, `linux-aarch64`, `macos-x86_64`, `macos-arm64`), one
   per glibc-hwcaps level actually needed (baseline `x86-64`, per the
   existing `-march=x86-64` override for these two files - Enzyme does not
   support the AVX mask intrinsics used for `x86-64-v3`, so there is only
   one variant per arch regardless of `VMECPP_HWCAPS_DISPATCH`). Compile
   inside the same container `cibuildwheel` will use
   (`manylinux_2_28`/`musllinux` image, or an equivalent macOS SDK/deployment
   target) so the resulting `.o` matches the wheel's glibc/SDK floor, and
   with matching `-fPIC`/ABI flags (Itanium C++ ABI, default libstdc++ on
   Linux and libc++ on macOS) so it links cleanly into a GCC/Apple-clang
   -built `vmecpp_core`.
   - Cache these objects (they only change when `exact_force_jvp.cc`,
     `exact_force_vjp.cc`, or the pinned Enzyme/LLVM versions change)
     and publish them as a build artifact or a small release asset,
     analogous to how `test_enzyme.yaml` already caches the
     `ClangEnzyme` plugin build.
2. **Feed the objects into the normal wheel build** via
   `[tool.cibuildwheel.linux.environment]` / `.macos.environment` /
   `.windows.environment`, e.g.
   `CMAKE_ARGS = "-DVMECPP_ENABLE_ENZYME=ON -DVMECPP_ENZYME_JVP_OBJECT=... -DVMECPP_ENZYME_VJP_OBJECT=..."`,
   with `before-build` fetching the matching prebuilt pair for the
   platform/arch being built (`cibuildwheel` already runs one container
   per arch, so `before-build` can `curl`/unpack the right pair using
   `$CIBW_ARCHS`/`uname -m`). No Clang, LLVM, or `ClangEnzyme` needs to be
   installed inside the `cibuildwheel` container itself; `before-build`
   only fetches two small `.o` files.
   - `vmecpp_core` and `vmecpp_core_v3` continue to be compiled with
     today's compiler and flags for every translation unit except those
     two; nothing about the default, non-adjoint code path changes.
3. **Fail closed, not silently, when a pair is missing** for a given
   target (e.g. a new arch added to the wheel matrix before its objects
   are built): leave `VMECPP_ENABLE_ENZYME` off for that specific build
   rather than failing the whole wheel job, and let the existing runtime
   guard (the "non-Enzyme build raises instead of using finite
   differences" behavior from #710) tell users why the exact-VJP API is
   unavailable on that wheel.

This keeps the wheel-build containers untouched for the code path every
non-adjoint user exercises, and confines all the toolchain risk to one
small, cacheable, independently-testable CI job.

## Verifying no performance regression

Because the only two translation units touched are `exact_force_jvp.cc`
and `exact_force_vjp.cc` - which are otherwise not referenced by the
production forward solve - object-identity is enough for everything else,
but should still be checked explicitly:
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
- Run this comparison against an actual wheel built through the proposed
  `cibuildwheel` path (not just the local CMake build used by
  `test_enzyme.yaml`), since the wheel is the artifact users install.

## Open questions

- Which wheel targets get Enzyme objects at all. `aarch64` and `musllinux`
  add real CI matrix cost for the object-build job; starting with
  `linux-x86_64` and both macOS architectures (matching where
  `test_enzyme.yaml` and the exact-HVP benchmark already run) and
  expanding once the mechanism is proven is likely the lowest-risk order.
- Whether prebuilt objects should be committed as versioned release
  assets (rebuilt on a schedule/on relevant source changes) or built
  fresh in every wheel release; the former is cheaper per release but
  needs its own staleness check against `exact_force_jvp.cc` /
  `exact_force_vjp.cc` content hashes and the pinned `ENZYME_REF`.
- Windows is currently outside `test_enzyme.yaml`'s scope entirely (no
  pinned Clang/Enzyme recipe exists for it yet); this proposal does not
  attempt to extend Enzyme support there, only to package what already
  exists for Linux/macOS.
