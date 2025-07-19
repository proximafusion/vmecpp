# Fourier Transform Debug Analysis

## Current Test Failures (6/8 tests failing)

1. **FourierToReal3DAsymmSingleMode**: Expected value ~1.14, got ~1.11 (2.5% error)
2. **RealToFourier3DAsymmSingleMode**: Round-trip coefficients wrong by ~0.03
3. **RoundTripTransform**: Similar round-trip accuracy issues
4. **FourierToReal2DAsymmSingleMode**: 2D version failing
5. **RealToFourier2DAsymmSingleMode**: 2D inverse failing
6. **NegativeNModeHandling**: Negative mode (m=1, n=-1) returns 0 instead of 1

## Key Issues Identified

### 1. Forward Transform (FourierToReal) Issues
- Values computed but with ~3% precision errors
- Indicates normalization or basis function problems
- Both 2D and 3D variants affected

### 2. Inverse Transform (RealToFourier) Issues
- Round-trip accuracy failing at 1e-10 precision level
- Suggests inverse normalization doesn't match forward
- Both symmetric and asymmetric coefficients affected

### 3. Negative Mode Handling Broken
- n=-1 modes return zero coefficients
- Critical for 3D stellarator configurations
- Likely sign convention or indexing issue

## Root Cause Categories

### A. Normalization Factors
- Forward transform uses sqrt(2) for m>0, n>0
- Inverse transform may have mismatched normalization
- Basis functions vs direct trigonometric inconsistency

### B. Basis Function vs Direct Trigonometry
- Forward uses FourierBasisFastPoloidal with built-in normalization
- Inverse uses direct cos/sin without normalization factors
- Mismatch between approaches

### C. Reflection and Symmetry Logic
- Complex asymmetric reflection for theta=[pi,2pi] range
- Potential sign errors in reflection formulas
- May affect round-trip consistency

## Next Steps (TDD Approach)

1. **Fix forward transform first** - make single mode tests pass
2. **Fix normalization consistency** - ensure forward/inverse match
3. **Fix negative mode handling** - critical for 3D cases
4. **Add more granular unit tests** - isolate specific failure modes
5. **Verify against analytical solutions** - known cos/sin patterns

## Priority Order

1. FourierToReal3DAsymmSingleMode (basic forward transform)
2. NegativeNModeHandling (critical for 3D)
3. RealToFourier3DAsymmSingleMode (round-trip consistency)
4. 2D variants (simpler cases)
