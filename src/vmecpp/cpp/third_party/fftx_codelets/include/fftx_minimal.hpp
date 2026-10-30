// Minimal FFTX type definitions used by vmecpp's vendored codelets.
//
// The full SPIRAL-generated FFTX library ships a much larger `fftx.hpp` and
// `fftx_base_types.hpp` beyond what vmecpp uses. We only consume:
//   * fftx::point_t<DIM>            -- {int x[DIM]} with operator[]
//   * transformTuple_t              -- declared in <kind>_cpu_public.h
//
// This header provides just `fftx::point_t<DIM>` so the vendored
// `fftx_<kind>_cpu_libentry.cpp` and `fftx_<kind>_cpu_public.h` continue to
// compile unchanged.  Everything else needed by SPIRAL's generated code lives
// in `omega64.h` (cospi/sinpi helpers).
//
// If you want to track upstream FFTX header changes, regenerate by re-running
// SPIRAL on this directory's `codegen/` inputs and copy the headers verbatim;
// no need to touch this file.

#ifndef VMECPP_THIRD_PARTY_FFTX_CODELETS_FFTX_MINIMAL_HPP_
#define VMECPP_THIRD_PARTY_FFTX_CODELETS_FFTX_MINIMAL_HPP_

namespace fftx {

template <int DIM>
struct point_t {
  int x[DIM];
  int operator[](unsigned char i) const { return x[i]; }
  int& operator[](unsigned char i) { return x[i]; }
};

}  // namespace fftx

#endif  // VMECPP_THIRD_PARTY_FFTX_CODELETS_FFTX_MINIMAL_HPP_
