"""Inputs, differentiable objectives, and small full-state exact Jacobians.

Standalone scripts can add the repository's ``tests`` directory to ``sys.path``
and import ``adjoint_verification.harness``; this module does not import pytest.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import autodiff, autodiff_wout, geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore

TEST_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
)
INPUT_NAMES = (
    "solovev",
    "solovev_analytical",
    "circular_tokamak",
    "cth_like_fixed_bdy",
    "cth_like_fixed_bdy_iota",
    "li383_low_res",
    "near_axis_iota_nfp4",
)
MPOL_VALUES = (4, 8, 12)
NS_VALUES = (11, 51, 101)


def load_input(
    name: str = "cth_like_fixed_bdy_iota",
    *,
    ns: int | None = None,
    mpol: int | None = None,
    ntor: int | None = None,
    ncurr: Literal[0, 1] | None = None,
    ftol: float = 1.0e-13,
    niter: int = 20000,
) -> vmecpp.VmecInput:
    """Return an independent symmetric, fixed-boundary, pinned-gauge input.

    Source profiles and multigrid schedules are preserved unless overridden.
    ``ns`` selects a single radial stage. Fourier resizing uses VmecInput.resize.
    """
    if name not in INPUT_NAMES:
        message = f"unknown verification input: {name}"
        raise ValueError(message)
    if ns is not None and ns < 3:
        message = "ns must be at least 3"
        raise ValueError(message)
    if ftol <= 0 or not np.isfinite(ftol) or niter < 1:
        message = "ftol must be finite and positive and niter must be positive"
        raise ValueError(message)
    source = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{name}.json")
    result = source.model_copy(deep=True)
    result.lasym = False
    result.lfreeb = False
    result.always_fix_m1_gauge = True
    if mpol is not None or ntor is not None:
        result.resize(
            result.mpol_max if mpol is None else mpol,
            result.ntor_max if ntor is None else ntor,
        )
    if ns is not None:
        result.ns_array = np.asarray([ns], dtype=np.int64)
    stages = len(result.ns_array)
    result.ftol_array = np.full(stages, ftol, dtype=np.float64)
    result.niter_array = np.full(stages, niter, dtype=np.int64)
    if ncurr is not None:
        result.ncurr = ncurr
    return result


def ncurr1_input(**kwargs) -> vmecpp.VmecInput:
    """Prescribed-current CTH input, including its original current profile."""
    return load_input("cth_like_fixed_bdy", ncurr=1, **kwargs)


def small_input(**kwargs) -> vmecpp.VmecInput:
    """The ns=7, mpol=4, ntor=2 CTH prescribed-current dense-test case."""
    return ncurr1_input(ns=7, mpol=4, ntor=2, **kwargs)


def resolution_input(
    mpol: int,
    ns: int,
    *,
    name: str = "cth_like_fixed_bdy_iota",
    **kwargs,
) -> vmecpp.VmecInput:
    """Independent member of the mpol={4,8,12}, ns={11,51,101} grid."""
    if mpol not in MPOL_VALUES or ns not in NS_VALUES:
        message = "resolution must belong to MPOL_VALUES x NS_VALUES"
        raise ValueError(message)
    return load_input(name, mpol=mpol, ns=ns, **kwargs)


def boundary_array(vmec_input: vmecpp.VmecInput) -> np.ndarray:
    """Copy the symmetric combined-basis boundary for autodiff.make_solver."""
    return np.stack((vmec_input.rbc, vmec_input.zbs)).astype(np.float64)


def solved_model(vmec_input: vmecpp.VmecInput):
    """Run the shared multigrid driver, requiring final convergence."""
    return autodiff._solve_model(
        vmec_input._to_cpp_vmecindata(), boundary_array(vmec_input)
    )


@dataclasses.dataclass(frozen=True)
class InvariantObjective:
    """A wout invariant: volume, aspect, iota(s), or Mercier DWell(s).

    ``s`` is a fixed normalized toroidal-flux coordinate, not a radial index.
    The discrete wout diagnostics retain their native discretization errors.
    """

    vmec_input: vmecpp.VmecInput
    quantity: str = "volume"
    s: float = 0.5

    def __post_init__(self):
        if self.quantity not in ("volume", "aspect", "iota", "well"):
            message = f"unknown invariant objective: {self.quantity}"
            raise ValueError(message)
        if not 0.0 < self.s < 1.0:
            message = "s must lie strictly inside (0, 1)"
            raise ValueError(message)
        object.__setattr__(self, "vmec_input", self.vmec_input.model_copy(deep=True))

    def __call__(self, value: geometry.Geometry):
        if self.quantity == "iota":
            jet = geometry.evaluate(value, jnp.asarray([self.s, 0.0, 0.0]))
            return jet[4, 1] / jet[3, 1]
        fields = autodiff_wout.wout_quantities(value, self.vmec_input)
        if self.quantity == "well":
            well = fields["DWell"]
            assert well is not None
            return jnp.interp(self.s, jnp.linspace(0.0, 1.0, len(well)), well)
        result = fields[self.quantity]
        assert result is not None
        return result


@dataclasses.dataclass(frozen=True)
class QuasisymmetryObjective:
    """The physical QS ratio residual squared, from the geometry field jets."""

    surfaces: tuple[float, ...] = (0.5,)
    helicity_m: int = 1
    helicity_n: int = 0
    ntheta: int = 16
    nphi: int = 16

    def __post_init__(self):
        if not self.surfaces or any(not 0.0 < s < 1.0 for s in self.surfaces):
            message = "QS surfaces must lie strictly inside (0, 1)"
            raise ValueError(message)
        if self.ntheta < 2 or self.nphi < 2:
            message = "QS angular grids must contain at least two points"
            raise ValueError(message)

    def residuals(self, value: geometry.Geometry):
        return qs.quasisymmetry_residuals(
            value,
            self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )

    def __call__(self, value: geometry.Geometry):
        return jnp.sum(self.residuals(value) ** 2)


@dataclasses.dataclass(frozen=True)
class BoundaryObjective:
    """Squared R/Z boundary coefficients; no interior or lambda dependence."""

    def __call__(self, value: geometry.Geometry):
        return sum(
            jnp.sum(getattr(value, name)[-1] ** 2)
            for name in ("r_cc", "r_ss", "r_sc", "r_cs", "z_sc", "z_cs", "z_cc", "z_ss")
        )


@dataclasses.dataclass(frozen=True)
class InteriorLambdaObjective:
    """Deliberately relabeling-dependent squared interior lambda_sc."""

    def __call__(self, value: geometry.Geometry):
        return jnp.sum(value.lambda_sc[1:-1] ** 2)


@dataclasses.dataclass(frozen=True)
class InteriorM2RObjective:
    """Deliberately relabeling-dependent squared interior m=2 R coefficients."""

    def __call__(self, value: geometry.Geometry):
        if value.r_cc.shape[1] <= 2:
            message = "the m=2 objective requires mpol >= 3"
            raise ValueError(message)
        return sum(
            jnp.sum(getattr(value, name)[1:-1, 2, :] ** 2)
            for name in ("r_cc", "r_ss", "r_sc", "r_cs")
        )


@dataclasses.dataclass(frozen=True)
class RandomLinearObjective:
    """Seeded linear functional of every Geometry leaf, including fluxes."""

    weights: geometry.Geometry

    @classmethod
    def from_geometry(cls, value: geometry.Geometry, seed: int = 0):
        rng = np.random.default_rng(seed)
        weights = jax.tree_util.tree_map(
            lambda leaf: jnp.asarray(rng.standard_normal(leaf.shape), dtype=leaf.dtype),
            value,
        )
        return cls(weights)

    def __call__(self, value: geometry.Geometry):
        products = jax.tree_util.tree_map(jnp.vdot, self.weights, value)
        return sum(jax.tree_util.tree_leaves(products))


def objective_suite(vmec_input: vmecpp.VmecInput, value: geometry.Geometry, seed=0):
    """Named scalar objectives shared by tests and standalone studies."""
    return {
        **{
            name: InvariantObjective(vmec_input, name)
            for name in ("volume", "aspect", "iota", "well")
        },
        "qs": QuasisymmetryObjective(),
        "boundary": BoundaryObjective(),
        "interior_lambda": InteriorLambdaObjective(),
        "interior_m2_r": InteriorM2RObjective(),
        "random_linear": RandomLinearObjective.from_geometry(value, seed),
    }


@dataclasses.dataclass(frozen=True)
class EvaluationSettings:
    """Arguments used to prepare the exact raw-force linearization."""

    iter1: int = 2
    iter2: int = 2
    precondition: bool = True
    always_fix_m1_gauge: bool = True

    def evaluate(self, model):
        model.evaluate(
            self.iter1, self.iter2, self.precondition, self.always_fix_m1_gauge
        )


@functools.lru_cache(maxsize=8)
def _binary_digest(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_fingerprint() -> dict[str, Any]:
    binary = Path(_vmecpp.__file__).resolve()
    stat = binary.stat()
    return {
        "binary": _binary_digest(str(binary), stat.st_size, stat.st_mtime_ns),
        "harness": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__,
    }


def _cache_metadata(model, vmec_input, state, evaluation, build_id):
    return json.dumps(
        {
            "schema": 1,
            "input": json.loads(vmec_input.to_json()),
            "state": hashlib.sha256(state.tobytes()).hexdigest(),
            "shape": list(state.shape),
            "dtype": state.dtype.str,
            "evaluation": dataclasses.asdict(evaluation),
            "build": _build_fingerprint(),
            "extra_build_id": build_id,
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_settings": {
                name: getattr(model, name, None)
                for name in (
                    "ns",
                    "mpol",
                    "ntor",
                    "mpol_geometry",
                    "ntor_geometry",
                    "lasym",
                    "lthreed",
                    "delt",
                    "iteration_style",
                    "always_fix_m1_gauge",
                    "have_to_flip_theta",
                )
            },
            "forces": hashlib.sha256(
                np.asarray(model.get_forces(), dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        sort_keys=True,
        allow_nan=False,
    )


def _validate_matrix(matrix, model, size):
    if matrix.shape != (size, size) or matrix.dtype != np.dtype(np.float64):
        return False
    if not np.isfinite(matrix).all():
        return False
    # Probe the full state to detect changed hidden linearization context.
    rng = np.random.default_rng(0)
    for _ in range(2):
        direction = np.ascontiguousarray(rng.standard_normal(size))
        actual = np.asarray(model.exact_hessian_vector_product(direction))
        if not np.allclose(matrix @ direction, actual, rtol=1.0e-10, atol=1.0e-12):
            return False
    return True


def _load_cached_matrix(path, metadata, model, size):
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > (
                size * size * 8 + len(metadata.encode()) * 4 + 65536
            ):
                return None
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["metadata"].item()) != metadata:
                return None
            matrix = cached["matrix"]
            checksum = hashlib.sha256(matrix.tobytes()).hexdigest()
            if checksum != str(cached["checksum"].item()):
                return None
        if _validate_matrix(matrix, model, size):
            return matrix
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    return None


def dense_jacobian(
    model,
    *,
    vmec_input: vmecpp.VmecInput | None = None,
    cache_dir: str | Path | None = None,
    evaluation: EvaluationSettings | None = None,
    build_id: str | None = None,
    max_state_size: int = 2048,
    max_bytes: int = 64 * 1024 * 1024,
) -> np.ndarray:
    """Assemble the full raw-force Jacobian using exact unit-vector HVPs only.

    Boundary, axis, structural-null, and gauge entries are not removed. The
    matrix is not symmetrized. Size limits apply before any quadratic allocation
    or cache loading. No finite-difference fallback exists.

    Caching requires the full input used to create the model, including profile
    overrides. Keys include its serialization, the full state, the compiled
    extension's content, harness version, and evaluation/model settings. Cache
    hits also require a checksum and two fresh exact full-state HVP probes.
    Invalid or stale entries are rebuilt. ``build_id`` adds a caller fingerprint
    (e.g. for dummy models); it does not replace the compiled-build fingerprint.
    """
    state = np.array(model.get_state(), dtype=np.float64, copy=True)
    size = state.size
    if state.ndim != 1 or size == 0 or not np.isfinite(state).all():
        message = "model state must be a nonempty finite flat vector"
        raise ValueError(message)
    if size > max_state_size or size * size * 8 > max_bytes:
        message = f"dense Jacobian of size {size} exceeds configured size limits"
        raise ValueError(message)
    if not model.has_exact_force_jacobian:
        message = "dense_jacobian requires an Enzyme-enabled exact HVP"
        raise RuntimeError(message)
    if cache_dir is not None and vmec_input is None:
        message = "cache_dir requires the full vmec_input used to create the model"
        raise ValueError(message)
    evaluation = EvaluationSettings() if evaluation is None else evaluation
    if not evaluation.always_fix_m1_gauge:
        message = "dense_jacobian requires always_fix_m1_gauge=True"
        raise ValueError(message)
    evaluation.evaluate(model)
    metadata = None
    path = None
    if cache_dir is not None:
        metadata = _cache_metadata(model, vmec_input, state, evaluation, build_id)
        key = hashlib.sha256(metadata.encode()).hexdigest()
        path = Path(cache_dir) / f"jacobian-{key}.npz"
        if path.exists():
            cached = _load_cached_matrix(path, metadata, model, size)
            if cached is not None:
                np.testing.assert_array_equal(np.asarray(model.get_state()), state)
                return cached
    matrix = np.empty((size, size), dtype=np.float64)
    direction = np.zeros(size, dtype=np.float64)
    for column in range(size):
        direction[column] = 1.0
        product = np.asarray(model.exact_hessian_vector_product(direction))
        if product.shape != (size,) or not np.isfinite(product).all():
            message = "exact HVP returned a nonfinite or incorrectly shaped vector"
            raise ValueError(message)
        matrix[:, column] = product
        direction[column] = 0.0
    np.testing.assert_array_equal(np.asarray(model.get_state()), state)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256(matrix.tobytes()).hexdigest()
        np.savez(path, matrix=matrix, metadata=metadata, checksum=checksum)
    return matrix
