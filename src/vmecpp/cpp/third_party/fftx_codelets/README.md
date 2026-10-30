# Vendored FFTX/SPIRAL codelets

This directory holds **pre-generated** FFTX/SPIRAL kernels that vmecpp uses
as its toroidal FFT backend (see `vmecpp/vmec/ideal_mhd_model/fft_toroidal.cc`).
The kernels are vendored so vmecpp builds without requiring users to install
the SPIRAL toolchain (OCaml + GAP + an OS-specific patched fork of
`spiral-software`).

## Layout

```
fftx_codelets/
├── README.md                            -- this file
├── codegen/                             -- inputs to reproduce the kernels
│   ├── dftbatch-sizes.txt               -- which (fftlen, nbatch) shapes
│   ├── fftx_prdftbat-frame.g            -- patched SPIRAL frame (fold nscale)
│   └── post_process.py                  -- post-codegen patches
├── include/
│   ├── omega64.h                        -- cospi/sinpi helpers (from SPIRAL)
│   └── fftx_minimal.hpp                 -- tiny replacement for `fftx.hpp`
├── lib_fftx_iprdftbat_cpu_srcs/         -- batched IPRDFT (c2r)
│   ├── fftx_iprdftbat_<N>_bat_<B>_APar_APar_CPU.cpp
│   ├── fftx_iprdftbat_cpu_libentry.cpp
│   ├── fftx_iprdftbat_cpu_metadata.cpp
│   ├── fftx_iprdftbat_cpu_decls.h
│   └── fftx_iprdftbat_cpu_public.h
└── lib_fftx_prdftbat_cpu_srcs/          -- batched PRDFT (r2c)
    └── ... (mirror, with vmecpp's `nscale[n]` fold baked in via SPIRAL spec)
```

The forward (PRDFT, r2c) kernels have a per-bin diagonal scaling fold of
`nscale[n] = 1 (n=0), sqrt(2) (n>=1)` baked into them at codegen time.  This
is the multiplier that vmecpp's spectral basis applies after the analysis
transform, so folding it into the FFT kernel saves the explicit post-multiply
pass; see `fft_toroidal.cc` accumulate loops for context.  The inverse
(IPRDFT, c2r) kernels are unscaled.

## How vmecpp uses them

`vmecpp_core` links these as a static library (`fftx_codelets`) and calls
`fftx_iprdftbat_cpu_Tuple` / `fftx_prdftbat_cpu_Tuple` to obtain
`{init, run, destroy}` function pointers for a given `(fftlen, nbatch,
read_stride=APar=0, write_stride=APar=0)` shape.

The shapes covered are vmecpp's default `nZeta = 2*ntor + 4` rule for
**even** `ntor` in `[6, 18]`, crossed with **even** `mpol` in `[6, 18]`
(this is a binary-size compromise; tweak the table to extend):

| `ntor` (-> `nZeta`)   | `mpol` (-> `nbatch = 12*mpol`)                              |
|-----------------------|-------------------------------------------------------------|
| 6, 8, 10, 12, 14, 16, 18 | 6, 8, 10, 12, 14, 16, 18                              |
| `nZeta` ∈ {16, 20, 24, 28, 32, 36, 40} | `nbatch` ∈ {72, 96, 120, 144, 168, 192, 216} |

Total: 7 × 7 = 49 forward + 49 inverse codelets, ~2.3 MB of source.

When a configuration falls outside this table, `ToroidalFftPlans` reports
that no kernel was found and `IdealMhdModel` runs the partial-DFT path
(`FourierToReal3DSymmFastPoloidal` / `ForcesToFourier3DSymmFastPoloidal` in
`dft_toroidal.cc`) instead.  Look for the construction-time log line:

```
INFO  Toroidal FFT backend: FFTX (nZeta=36, 12*mpol=144).
```
or
```
WARN  Toroidal FFT backend: partial-DFT fallback (no FFTX kernel for nZeta=50, 12*mpol=120). ...
```

To extend the table, edit `codegen/dftbatch-sizes.txt` and regenerate (see
below).

## Regenerating the codelets

Required tools (one-time setup, only when you want to regenerate -- end-users
of vmecpp do **not** need any of this):

* OCaml 4.x or 5.x (`ocaml`, `ocamlfind`, `ocamlbuild`) -- via apt or opam
* GNU autotools (`autoconf`, `automake`, `libtool`) -- only if you intend to
  rebuild SPIRAL itself
* Python 3.6+
* C/C++ compiler with C99 support
* SPIRAL 8.5+ from <https://github.com/spiral-software/spiral-software>
* FFTX from <https://github.com/spiral-software/fftx>

SPIRAL on modern Linux (glibc >= 2.41, gcc >= 14) needs a few patches we have
documented in our session notes -- not in upstream as of writing.  The
short list:

* `gap/src/system.c`: replace `<termio.h>` with `<termios.h>` plus
  `<sys/ioctl.h>` and `<unistd.h>`; `TCGETA`/`TCSETAW` -> `TCGETS`/`TCSETSW`.
* `profiler/targets/include/omega64.h`: rename `cospi`/`sinpi` to
  `spiral_cospi`/`spiral_sinpi` and add `#define`s, to avoid clashing with
  glibc 2.41+ math intrinsics of the same name.
* `gap/src/CMakeLists.txt` (or `CFLAGS`): build with `-std=gnu99` so the K&R
  function definitions in `gap/src/md5.c` and friends parse.

### Steps to regenerate

```bash
# 1. Install SPIRAL & FFTX once (one-time, see patches above).
export SPIRAL_HOME=$HOME/src/fftx/spiral-software
export FFTX_HOME=$HOME/src/fftx/fftx

# 2. Configure FFTX with our size table & patched forward frame.
cp codegen/dftbatch-sizes.txt    "$FFTX_HOME/src/library/dftbatch-sizes.txt"
cp codegen/fftx_prdftbat-frame.g "$FFTX_HOME/src/library/fftx_prdftbat-frame.g"

# 3. Run the SPIRAL/FFTX codegen for both directions.
cd "$FFTX_HOME/src/library"
rm -rf lib_fftx_iprdftbat_cpu_srcs lib_fftx_prdftbat_cpu_srcs
python3 gen_dftbat.py -t fftx_prdftbat -s dftbatch-sizes.txt -p CPU       # forward (PRDFT, r2c)
python3 gen_dftbat.py -t fftx_prdftbat -s dftbatch-sizes.txt -p CPU -i    # inverse (IPRDFT, c2r)

# 4. Apply our post-processing fixes (compound-literal C bug + thread_local
#    workspaces).
python3 path/to/vmecpp/src/vmecpp/cpp/third_party/fftx_codelets/codegen/post_process.py \
    lib_fftx_iprdftbat_cpu_srcs lib_fftx_prdftbat_cpu_srcs

# 5. Replace the vendored copies inside vmecpp.
cp lib_fftx_iprdftbat_cpu_srcs/* path/to/vmecpp/src/vmecpp/cpp/third_party/fftx_codelets/lib_fftx_iprdftbat_cpu_srcs/
cp lib_fftx_prdftbat_cpu_srcs/*  path/to/vmecpp/src/vmecpp/cpp/third_party/fftx_codelets/lib_fftx_prdftbat_cpu_srcs/

# 6. Re-apply the local-headers patch (so kernels include "fftx_minimal.hpp"
#    instead of upstream "fftx.hpp"):
sed -i 's|#include "fftx.hpp"|#include "fftx_minimal.hpp"|' \
    path/to/vmecpp/src/vmecpp/cpp/third_party/fftx_codelets/lib_fftx_*_cpu_srcs/*.h
```

That's it.  Rebuild vmecpp normally afterwards.

## Why the post-processing step exists

SPIRAL is otherwise excellent but has two CPU-codegen bugs we hit when using
the small-Diag formulation (`RCDiag(FList(TReal, [s_0, 0, s_1, 0, ...]))`)
that lets it keep the batch loop rolled instead of unrolling 75x:

1. For some sizes (n>=36) it emits the diagonal as an inline anonymous array
   `(({...constants...} + offset))`, which is GCC statement-expression syntax,
   not a valid compound literal.  The fix is mechanical: prefix with
   `(const double[])` so it becomes a proper C99 compound literal.
2. Workspace arrays inside the run-loop are emitted as plain `static`, so
   concurrent calls from multiple OpenMP threads alias.  We promote them to
   `static thread_local`.  Twiddle tables at file scope (set by `init_*` and
   read-only afterwards) deliberately stay plain `static` so we get one shared
   copy.

Both transforms are applied by `codegen/post_process.py`.

## License

FFTX and SPIRAL are BSD-licensed
(<https://github.com/spiral-software/fftx/blob/main/License.txt>,
<https://github.com/spiral-software/spiral-software/blob/master/LICENSE>).
The vendored kernels in this directory inherit that BSD license.  See the
copyright headers on the individual files.
