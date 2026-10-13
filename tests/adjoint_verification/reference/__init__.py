"""WP6 interim references: converged native VMEC, never experimental Newton."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import autodiff, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

from ..harness import boundary_array, load_input, objective_suite, small_input

OBJECTIVES = ("volume", "aspect", "iota", "well", "qs", "boundary", "random_linear")
DEFAULT_STEPS = np.geomspace(1.0e-2, 1.0e-7, 11)
COMPONENT_STEPS = np.asarray([1.0e-3, 3.0e-4, 1.0e-4])


def _checkpoint_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_checkpoint(path, arrays):
    """Replace a checkpoint only after its same-directory NPZ is durable."""
    pending = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with pending.open("xb") as stream:
            np.savez(stream, **arrays, checksum=np.asarray(_checkpoint_digest(arrays)))
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        pending.unlink(missing_ok=True)


def _read_checkpoint(path, metadata, shape):
    try:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        checksum = arrays.pop("checksum")
        if checksum.shape != () or str(checksum) != _checkpoint_digest(arrays):
            message = "checksum mismatch"
            raise ValueError(message)
        expected = {*metadata, "plus", "minus", "completed", "records", "failures"}
        if set(arrays) != expected:
            message = "incompatible checkpoint fields"
            raise ValueError(message)
        for name, value in metadata.items():
            cached = arrays[name]
            if (
                cached.dtype != value.dtype
                or cached.shape != value.shape
                or not np.array_equal(cached, value)
            ):
                message = f"incompatible {name}"
                raise ValueError(message)
        for name in ("plus", "minus"):
            if arrays[name].shape != shape or arrays[name].dtype != np.dtype(float):
                message = f"invalid {name} shape or dtype"
                raise ValueError(message)
        if (
            arrays["completed"].shape != shape[1:]
            or arrays["completed"].dtype != np.dtype(bool)
            or arrays["records"].ndim != 2
            or arrays["records"].shape[1] != 5
            or arrays["records"].dtype != np.dtype(float)
            or arrays["failures"].shape != (len(arrays["records"]),)
            or arrays["failures"].dtype.kind != "U"
        ):
            message = "invalid completion mask or solve records"
            raise ValueError(message)
        return arrays
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, TypeError) as error:
        message = f"invalid component checkpoint {path}: {error}"
        raise ValueError(message) from error


def reference_input(name="small"):
    """Use the shared WP0 configuration, with ftol=1e-16 at every stage."""
    return (
        small_input(ftol=1.0e-16) if name == "small" else load_input(name, ftol=1.0e-16)
    )


def native_restart(vmec_input, boundary=None):
    """Build the native restart state from a full pinned-gauge VMEC run."""
    vmec_input = vmec_input.model_copy(deep=True)
    vmec_input.ftol_array = np.full(len(vmec_input.ns_array), 1.0e-16)
    vmec_input.always_fix_m1_gauge = True
    if boundary is not None:
        vmec_input.rbc = np.asarray(boundary[0]).copy()
        vmec_input.zbs = np.asarray(boundary[1]).copy()
    output = vmecpp.run(vmec_input, max_threads=1, verbose=False)
    return _vmecpp.HotRestartState(
        wout=output.wout._to_cpp_wout(),
        indata=output.input._to_cpp_vmecindata(),
    )


def solve_reference(vmec_input, boundary=None, *, restart_from=None):
    """Independently solve a boundary, optionally from a native HotRestartState.

    Native restart initialization shifts interior surfaces for the new boundary. The
    pinned gauge is reset to the independent cold initialization at that boundary. Every
    result must meet native convergence.
    """
    vmec_input = vmec_input.model_copy(deep=True)
    vmec_input.ftol_array = np.full(len(vmec_input.ns_array), 1.0e-16)
    vmec_input.always_fix_m1_gauge = True
    boundary = boundary_array(vmec_input) if boundary is None else np.asarray(boundary)
    if (
        boundary.shape != boundary_array(vmec_input).shape
        or not np.isfinite(boundary).all()
    ):
        message = "boundary must be finite and match the input shape"
        raise ValueError(message)
    indata = vmec_input._to_cpp_vmecindata()
    indata.rbc[...] = boundary[0]
    indata.zbs[...] = boundary[1]
    resolutions = [int(ns) for ns in vmec_input.ns_array if ns >= 3]
    if not resolutions:
        message = "reference input needs a radial resolution >= 3"
        raise ValueError(message)
    if restart_from is None:
        model = _vmecpp.VmecModel.create(indata, resolutions[0])
        for index, ns in enumerate(resolutions):
            if index:
                model.refine_to(ns)
            model.solve()
    else:
        if not isinstance(restart_from, _vmecpp.HotRestartState):
            message = "restart_from must be a native HotRestartState"
            raise ValueError(message)
        cold = _vmecpp.VmecModel.create(indata, resolutions[-1])
        initial = np.asarray(cold.get_state()).copy()
        model = _vmecpp.VmecModel.create(
            indata, resolutions[-1], initial_state=restart_from
        )
        state = np.asarray(model.get_state()).copy()
        pinned = autodiff._gauge_entries(model)
        state[pinned] = initial[pinned]
        model.set_state(np.ascontiguousarray(state))
        model.solve()
    residuals = np.asarray([model.fsqr, model.fsqz, model.fsql])
    if (
        model.status != 11
        or model.ftolv > 1.0e-16
        or not np.isfinite(residuals).all()
        or np.any(residuals > model.ftolv)
    ):
        message = (
            f"reference did not converge: status={model.status}, "
            f"ftol={model.ftolv}, residuals={residuals.tolist()}"
        )
        raise RuntimeError(message)
    return model


def active_boundary(vmec_input):
    """Identify structurally inactive slots, retaining redundant signed-n slots."""
    mask = np.ones(boundary_array(vmec_input).shape, dtype=bool)
    ntor = vmec_input.ntor
    mask[1, 0, ntor] = False
    mpol_geometry = vmec_input.mpol_geometry
    if not 1 <= mpol_geometry <= vmec_input.mpol:
        mpol_geometry = vmec_input.mpol
    ntor_geometry = vmec_input.ntor_geometry
    if not 0 <= ntor_geometry <= vmec_input.ntor:
        ntor_geometry = vmec_input.ntor
    mask[:, mpol_geometry:, :] = False
    mask[:, :, np.abs(np.arange(-ntor, ntor + 1)) > ntor_geometry] = False
    return mask


def directions(vmec_input, seed=0):
    """Seeded unit directions with explicit inactive high-n on axisymmetry."""
    mask = active_boundary(vmec_input)
    # Use independent physical modes for directions, but scan every dense slot.
    mask[:, 0, : vmec_input.ntor] = False
    rng = np.random.default_rng(seed)
    result = {}
    for name in ("random", "m1", "highm", "highn"):
        selected = mask.copy()
        if name == "m1":
            selected[:, np.arange(vmec_input.mpol) != 1, :] = False
        elif name == "highm":
            selected[:, :-1, :] = False
        elif name == "highn":
            selected[
                :,
                :,
                np.abs(np.arange(-vmec_input.ntor, vmec_input.ntor + 1))
                != vmec_input.ntor,
            ] = False
            if vmec_input.ntor == 0:
                selected[:] = False
        direction = rng.standard_normal(mask.shape) * selected
        norm = np.linalg.norm(direction)
        result[name] = None if norm == 0 else direction / norm
    return result


def taylor_assessment(steps, remainder, noise_floor, *, min_points=4):
    """Measure the entire contiguous prefix before the measured noise floor.

    Require at least three adjacent slopes, all near two, plus a fitted slope near two.
    Zeros and the noise-floor tail are never logarithm arguments. An exactly affine
    objective is inconclusive rather than quadratic evidence.
    """
    steps, remainder = np.asarray(steps), np.asarray(remainder)
    if (
        steps.ndim != 1
        or remainder.shape != steps.shape
        or len(steps) < min_points
        or not np.isfinite(steps).all()
        or np.any(steps <= 0)
        or np.any(np.diff(steps) >= 0)
        or not np.isfinite(noise_floor)
        or noise_floor < 0
    ):
        message = "need decreasing positive steps and a finite nonnegative noise"
        raise ValueError(message)
    usable = np.isfinite(remainder) & (remainder > 10 * noise_floor) & (remainder > 0)
    stop = int(np.flatnonzero(~usable)[0]) if not usable.all() else len(steps)
    slopes = np.full(len(steps) - 1, np.nan)
    fitted = np.nan
    if stop >= 2:
        slopes[: stop - 1] = np.diff(np.log(remainder[:stop])) / np.diff(
            np.log(steps[:stop])
        )
        fitted = float(np.polyfit(np.log(steps[:stop]), np.log(remainder[:stop]), 1)[0])
    passed = bool(
        np.isfinite(remainder).all()
        and stop >= min_points
        and abs(fitted - 2) <= 0.2
        and np.all(np.abs(slopes[: stop - 1] - 2) <= 0.3)
    )
    return {
        "slopes": slopes,
        "fit_slope": fitted,
        "fit_stop": stop,
        "noise_floor": noise_floor,
        "passed": passed,
        "status": (
            "pass"
            if passed
            else (
                "noise"
                if stop < min_points and np.isfinite(remainder).all()
                else "failure"
            )
        ),
    }


def _steps(steps, min_points=4):
    steps = np.asarray(steps, dtype=np.float64)
    if (
        steps.ndim != 1
        or len(steps) < min_points
        or not np.isfinite(steps).all()
        or np.any(steps <= 0)
        or np.any(np.diff(steps) >= 0)
    ):
        message = f"at least {min_points} decreasing finite positive steps are required"
        raise ValueError(message)
    return steps


def _comparison(fd, ad, objective_noise, steps):
    absolute = np.abs(fd - ad)
    denominator = np.maximum(np.abs(fd), np.abs(ad))
    near_zero = denominator <= 1.0e-8
    relative = np.full_like(absolute, np.nan)
    np.divide(absolute, denominator, out=relative, where=~near_zero)
    noise = objective_noise / steps.reshape((-1,) + (1,) * (fd.ndim - 1))
    threshold = np.where(near_zero, 1.0e-8, 1.0e-6 * denominator)
    resolved = noise < threshold
    matches = resolved & (absolute <= threshold)
    return {
        "absolute_error": absolute,
        "relative_error": relative,
        "near_zero": near_zero,
        "derivative_noise": noise,
        "resolved": resolved,
        "acceptance_threshold": threshold,
        "acceptance_rule": "relative-only for nonzero gradients; absolute only when both are <=1e-8",
        "rtol": 1.0e-6,
        "atol": 1.0e-8,
    }, matches


def _constant_iota_assessment(
    vmec_input, objective, gradient, base_value, plus, minus, remainder
):
    """Verify the prescribed-iota boundary independence, not a Taylor slope."""
    if objective != "iota" or vmec_input.ncurr != 0:
        return None
    tolerance = 64 * np.finfo(float).eps * max(1, abs(base_value))
    value_error = np.maximum(np.abs(plus - base_value), np.abs(minus - base_value))
    gradient_tolerance = 1.0e-12
    passed = bool(
        np.isfinite(gradient).all()
        and np.max(np.abs(gradient)) <= gradient_tolerance
        and np.isfinite(value_error).all()
        and np.all(value_error <= tolerance)
        and np.isfinite(remainder).all()
        and np.all(remainder <= tolerance)
    )
    return {
        "slope_applicable": False,
        "fit_slope": np.nan,
        "fit_stop": 0,
        "slopes": np.full(len(remainder) - 1, np.nan),
        "constant_value_error": value_error,
        "constant_absolute_tolerance": float(tolerance),
        "constant_gradient_absolute_tolerance": gradient_tolerance,
        "constant_reason": "prescribed iota, ncurr=0, is boundary-independent",
        "passed": passed,
        "status": "constant" if passed else "failure",
    }


class ReferenceStudy:
    """Fixed base geometry/objectives, exact boundary VJP, independent FD solves."""

    def __init__(self, vmec_input, objectives=OBJECTIVES, *, seed=0, hot_restart=False):
        if not vmecpp.has_exact_force_jacobian():
            message = "WP6 gradients require an Enzyme-enabled runtime"
            raise RuntimeError(message)
        if not objectives or any(name not in OBJECTIVES for name in objectives):
            message = f"objectives must belong to {OBJECTIVES}"
            raise ValueError(message)
        jax.config.update("jax_enable_x64", True)
        self.input = vmec_input.model_copy(deep=True)
        self.input.ftol_array = np.full(len(self.input.ns_array), 1.0e-16)
        self.input.always_fix_m1_gauge = True
        self.boundary = boundary_array(self.input)
        self.model = solve_reference(self.input)
        self.base_geometry = geometry.from_cpp(self.model.get_geometry())
        suite = objective_suite(self.input, self.base_geometry, seed)
        self.objectives = {name: suite[name] for name in objectives}
        self.hot_restart = hot_restart
        self.seed = seed
        solver = autodiff.make_solver(self.input)
        self.gradients = {}
        self.values = {}
        self.noise = {}
        # Return from a nearby equilibrium so the noise probe takes a different
        # trajectory, rather than immediately accepting the identical base state.
        self.restart = native_restart(self.input)
        self.noise_probe_step = 1.0e-4
        self.noise_probe_direction = directions(self.input, seed)["random"]
        probe = native_restart(
            self.input,
            self.boundary + self.noise_probe_step * self.noise_probe_direction,
        )
        repeat = solve_reference(self.input, restart_from=probe)
        repeat_geometry = geometry.from_cpp(repeat.get_geometry())
        for name, objective in self.objectives.items():
            value, gradient = jax.value_and_grad(
                lambda b, obj=objective: obj(solver(b))
            )(jnp.asarray(self.boundary))
            self.values[name] = float(objective(self.base_geometry))
            self.gradients[name] = np.asarray(gradient)
            self.noise[name] = max(
                abs(float(objective(repeat_geometry)) - self.values[name]),
                abs(float(value) - self.values[name]),
                float(16 * np.finfo(float).eps * max(1, abs(self.values[name]))),
            )
            if not np.isfinite(self.gradients[name]).all():
                message = f"nonfinite exact gradient for {name}"
                raise RuntimeError(message)
        self.records = [
            [m.status, m.ftolv, m.fsqr, m.fsqz, m.fsql] for m in (self.model, repeat)
        ]
        self.failures = ["", ""]
        self._provenance = None

    def evaluate(self, boundary):
        """Evaluate all fixed objectives at one freshly solved equilibrium."""
        try:
            model = solve_reference(
                self.input,
                boundary,
                restart_from=self.restart if self.hot_restart else None,
            )
        except RuntimeError as error:
            self.records.append([-1, 1.0e-16, np.nan, np.nan, np.nan])
            self.failures.append(str(error))
            return np.full(len(self.objectives), np.nan)
        self.records.append(
            [model.status, model.ftolv, model.fsqr, model.fsqz, model.fsql]
        )
        self.failures.append("")
        value = geometry.from_cpp(model.get_geometry())
        return np.asarray([float(obj(value)) for obj in self.objectives.values()])

    def provenance(self):
        if self._provenance is not None:
            return self._provenance
        binary = Path(_vmecpp.__file__).resolve()
        self._provenance = json.dumps(
            {
                "reference": "native VmecModel.solve, interim ftol=1e-16 (not Newton)",
                "input": json.loads(self.input.to_json()),
                "seed": self.seed,
                "hot_restart": self.hot_restart,
                "restart_api": "HotRestartState from vmecpp.run; cold-initialized pinned gauge",
                "exact_gradient": "jax.value_and_grad(autodiff.make_solver)",
                "enzyme": vmecpp.has_exact_force_jacobian(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "jax": jax.__version__,
                "numpy": np.__version__,
                "noise": "base versus return from a nearby native equilibrium, plus roundoff",
                "noise_probe_step": self.noise_probe_step,
                "force_tolerance": "squared/normalized force threshold, not state accuracy",
                "failed_solve_status": -1,
            },
            sort_keys=True,
        )
        return self._provenance

    def taylor(self, steps=DEFAULT_STEPS, names=("random", "m1", "highm", "highn")):
        steps = _steps(steps)
        available = directions(self.input, self.seed)
        data = []
        for name in names:
            direction = available[name]
            if direction is None:
                data.append(
                    {"direction_name": name, "status": "inactive", "passed": False}
                )
                continue
            plus = np.asarray(
                [self.evaluate(self.boundary + h * direction) for h in steps]
            )
            minus = np.asarray(
                [self.evaluate(self.boundary - h * direction) for h in steps]
            )
            for column, objective in enumerate(self.objectives):
                derivative = float(np.vdot(self.gradients[objective], direction))
                central_fd = (plus[:, column] - minus[:, column]) / (2 * steps)
                comparison, matches = _comparison(
                    central_fd, derivative, self.noise[objective], steps
                )
                derivative_passed = bool(
                    np.isfinite(central_fd).all() and np.any(matches[:-1] & matches[1:])
                )
                remainder = np.abs(
                    plus[:, column] - self.values[objective] - steps * derivative
                )
                assessment = taylor_assessment(steps, remainder, self.noise[objective])
                assessment["slope_applicable"] = True
                failed_solve = (
                    not np.isfinite(plus[:, column]).all()
                    or not np.isfinite(minus[:, column]).all()
                )
                if failed_solve:
                    assessment["passed"] = False
                    assessment["status"] = "failure"
                elif assessment["passed"] and not derivative_passed:
                    assessment["passed"] = False
                    resolved = comparison["resolved"]
                    assessment["status"] = (
                        "failure"
                        if np.any(resolved[:-1] & resolved[1:])
                        else "unresolved"
                    )
                constant = _constant_iota_assessment(
                    self.input,
                    objective,
                    self.gradients[objective],
                    self.values[objective],
                    plus[:, column],
                    minus[:, column],
                    remainder,
                )
                if constant is not None:
                    assessment.update(constant)
                    derivative_passed = constant["passed"]
                data.append(
                    {
                        "objective": objective,
                        "direction_name": name,
                        "direction": direction,
                        "steps": steps,
                        "base_value": self.values[objective],
                        "plus_values": plus[:, column],
                        "minus_values": minus[:, column],
                        "central_fd": central_fd,
                        "directional_ad": derivative,
                        "derivative_passed": derivative_passed,
                        "failed_native_solve": failed_solve,
                        "remainder": remainder,
                        **comparison,
                        **assessment,
                    }
                )
        return data

    def _component_metadata(self, steps):
        names = list(self.objectives)
        metadata = {
            "schema": np.asarray(1),
            "shape": np.asarray(self.boundary.shape),
            "steps": steps,
            "boundary": np.asarray(self.boundary),
            "base_state": np.asarray(self.model.get_state()),
            "objective_names": np.asarray(names),
            "base_values": np.asarray([self.values[name] for name in names]),
            "noise": np.asarray([self.noise[name] for name in names]),
            "gradient_ad": np.asarray([self.gradients[name] for name in names]),
            "input": np.asarray(self.input.to_json()),
            "provenance": np.asarray(self.provenance()),
            "helper_sha256": np.asarray(
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ),
            "objective_helper_sha256": np.asarray(
                hashlib.sha256(
                    (Path(__file__).parents[1] / "harness" / "__init__.py").read_bytes()
                ).hexdigest()
            ),
        }
        for column, objective in enumerate(self.objectives.values()):
            for index, weight in enumerate(
                jax.tree_util.tree_leaves(getattr(objective, "weights", ()))
            ):
                metadata[f"weight_{column}_{index}"] = np.asarray(weight)
        metadata["provenance_sha256"] = np.asarray(_checkpoint_digest(metadata))
        return metadata

    def _checkpoint_records(self):
        return {
            "records": np.asarray(self.records, dtype=float).reshape(-1, 5),
            "failures": np.asarray(self.failures, dtype=str),
        }

    def components(self, steps=COMPONENT_STEPS, *, checkpoint=None, resume=False):
        """Scan all signed slots; optionally save each completed +/- pair.

        ``checkpoint`` is an NPZ path. ``resume=True`` requires a compatible
        checkpoint and skips only its completed pairs. New scans replace any
        existing checkpoint. Objective columns retain their exact input order.
        """
        steps = _steps(steps, min_points=3)
        if resume and checkpoint is None:
            message = "component resume requires a checkpoint path"
            raise ValueError(message)
        mask = active_boundary(self.input)
        shape = (len(self.objectives), len(steps), *self.boundary.shape)
        plus = np.full(shape, np.nan)
        minus = np.full(shape, np.nan)
        completed = np.zeros(shape[1:], dtype=bool)
        metadata = {}
        path = None if checkpoint is None else Path(checkpoint)
        if path is not None:
            metadata = self._component_metadata(steps)
            if resume:
                saved = _read_checkpoint(path, metadata, shape)
                plus, minus, completed = (
                    saved[name] for name in ("plus", "minus", "completed")
                )
                self.records = saved["records"].tolist()
                self.failures = saved["failures"].tolist()
            else:
                _write_checkpoint(
                    path,
                    {
                        **metadata,
                        "plus": plus,
                        "minus": minus,
                        "completed": completed,
                        **self._checkpoint_records(),
                    },
                )
        plus_flat = plus.reshape(len(self.objectives), len(steps), -1)
        minus_flat = minus.reshape(len(self.objectives), len(steps), -1)
        completed_flat = completed.reshape(len(steps), -1)
        for index in range(mask.size):
            direction = np.zeros_like(self.boundary)
            direction.ravel()[index] = 1
            for k, h in enumerate(steps):
                if completed_flat[k, index]:
                    continue
                p = self.evaluate(self.boundary + h * direction)
                m = self.evaluate(self.boundary - h * direction)
                plus_flat[:, k, index] = p
                minus_flat[:, k, index] = m
                completed_flat[k, index] = True
                if path is not None:
                    _write_checkpoint(
                        path,
                        {
                            **metadata,
                            "plus": plus,
                            "minus": minus,
                            "completed": completed,
                            **self._checkpoint_records(),
                        },
                    )
        fd = (plus - minus) / (2 * steps.reshape((-1,) + (1,) * self.boundary.ndim))
        results = []
        for column, name in enumerate(self.objectives):
            ad = self.gradients[name]
            comparison, matches = _comparison(fd[column], ad, self.noise[name], steps)
            scale = max(1, float(np.max(np.abs(ad))))
            # Require agreement at two adjacent resolved steps, not one lucky FD.
            passed_modes = np.any(matches[:-1] & matches[1:], axis=0)
            passed_modes &= np.isfinite(fd[column]).all(axis=0)
            inactive_zero = (
                (~mask)
                & (np.abs(ad) <= 1.0e-12 * scale)
                & np.all(np.abs(fd[column]) <= 1.0e-12 * scale, axis=0)
            )
            redundant = np.zeros_like(mask)
            redundant[:, 0, : self.input.ntor] = True
            passed_modes[~mask] = inactive_zero[~mask]
            resolved = comparison["resolved"]
            noise_limited = (
                mask
                & np.isfinite(fd[column]).all(axis=0)
                & ~np.any(resolved[:-1] & resolved[1:], axis=0)
            )
            passed = bool(passed_modes.all())
            status = (
                "pass"
                if passed
                else ("unresolved" if noise_limited[~passed_modes].all() else "failure")
            )
            results.append(
                {
                    "objective": name,
                    "steps": steps,
                    "gradient_ad": ad,
                    "gradient_fd": fd[column],
                    "plus_values": plus[column],
                    "minus_values": minus[column],
                    **comparison,
                    "active": mask,
                    "inactive_zero": inactive_zero,
                    "redundant": redundant,
                    "passed_modes": passed_modes,
                    "noise_limited": noise_limited,
                    "failed_native_solve": ~np.isfinite(fd[column]).all(axis=0),
                    "passed": passed,
                    "status": status,
                    "m": np.arange(self.input.mpol),
                    "n": np.arange(-self.input.ntor, self.input.ntor + 1),
                }
            )
        return results
