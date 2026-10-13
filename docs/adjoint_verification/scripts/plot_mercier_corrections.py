"""Experimental WP7 Mercier FD and residual-correction measurements.

Run with --cases small li383_low_res. Scientific measurements enable JAX x64. Results
and incremental numerical checkpoints are written only under
docs/adjoint_verification/results. This test-only experiment does not promote an adjoint
acceptance contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "docs" / "adjoint_verification" / "results"
TERMS = ("DMerc", "DCurr", "DGeod", "DWell")
STEPS = np.asarray([1e-3, 3e-4, 1e-4])
TOL = 1e-12
FD_RTOL = 1e-6


def enable_x64(config: Any) -> None:
    """Require double precision before constructing scientific JAX arrays."""
    config.update("jax_enable_x64", True)
    if not config.x64_enabled:
        message = "Mercier measurements require JAX x64; no float32 result is accepted"
        raise RuntimeError(message)


def checkpoint(data: dict[str, Any], output: Path, case: str) -> None:
    """Atomically preserve every completed measurement without pickle."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"plot_08_mercier_corrections_{case}.npz"
    pending = destination.with_suffix(".npz.part")
    arrays: dict[str, Any] = {name: np.asarray(value) for name, value in data.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        message = "checkpoint arrays must not require pickle"
        raise ValueError(message)
    with pending.open("wb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **arrays)
    pending.replace(destination)


def render(data: dict[str, Any], output: Path, case: str) -> None:
    """Render saved numerical records; this function requires no native runtime."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_style import STYLE  # noqa: PLC0415

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for index, term in enumerate(TERMS):
            axes[0, 0].loglog(
                data["steps"], data["fd_relative_error"][:, index], "o-", label=term
            )
            for state_index, marker in enumerate(("o", "s")):
                axes[0, 1].loglog(
                    data["correction_residual"][state_index, index],
                    data["gradient_relative_error"][state_index, index, 0],
                    marker,
                    color=f"C{index}",
                    label=f"{term} naive" if state_index == 0 else None,
                )
                axes[0, 1].loglog(
                    data["correction_residual"][state_index, index],
                    data["gradient_relative_error"][state_index, index, 1],
                    marker,
                    markerfacecolor="none",
                    color=f"C{index}",
                    label=f"{term} corrected" if state_index == 0 else None,
                )
            for method, style in enumerate(("-", "--")):
                axes[1, 0].loglog(
                    data["correction_residual"][:, index],
                    data["value_absolute_error"][:, index, method],
                    style,
                    color=f"C{index}",
                    label=f"{term} {'naive' if method == 0 else 'corrected'}",
                )
        axes[0, 0].axhline(FD_RTOL, color="black", linestyle="--", label="1e-6")
        axes[0, 0].set(xlabel="independent boundary step", ylabel="relative FD error")
        axes[0, 1].set(xlabel="active raw force norm", ylabel="boundary gradient error")
        axes[1, 0].set(xlabel="active raw force norm", ylabel="absolute value error")
        histories = sorted(name for name in data if name.endswith("_residual_history"))
        for name in histories:
            history = data[name]
            axes[1, 1].semilogy(
                np.arange(len(history)),
                history,
                ".-",
                label=name.removesuffix("_residual_history"),
            )
        axes[1, 1].axhline(TOL, color="black", linestyle="--", label="1e-12")
        axes[1, 1].set(xlabel="accepted Newton step", ylabel="active raw force norm")
        for axis in axes.flat:
            axis.legend(fontsize=6)
        figure.suptitle(f"{case}: {data['status']}; experimental WP7")
        figure.tight_layout()
        destination = output / f"plot_08_mercier_corrections_{case}.png"
        pending = destination.with_suffix(".png.part")
        try:
            figure.savefig(pending, format="png", dpi=160, bbox_inches="tight")
            pending.replace(destination)
        finally:
            plt.close(figure)


def initial_data(case: str) -> dict[str, Any]:
    return {
        "case": case,
        "status": "running",
        "failure": "",
        "terms": TERMS,
        "steps": STEPS,
        "newton_tol": TOL,
        "fd_rtol": FD_RTOL,
        "solve_status": np.full(7, "pending", dtype="U64"),
        "solve_labels": (
            "reference",
            "plus_0",
            "minus_0",
            "plus_1",
            "minus_1",
            "plus_2",
            "minus_2",
        ),
        "fd_status": np.full(4, "pending", dtype="U64"),
        "public_api_status": "pending",
        "correction_status": np.full((2, 4), "pending", dtype="U64"),
        "fd_values": np.full((3, 2, 4), np.nan),
        "fd_derivative": np.full((3, 4), np.nan),
        "fd_relative_error": np.full((3, 4), np.nan),
        "fd_plateau_relative_change": np.full(4, np.nan),
        "correction_residual": np.full((2, 4), np.nan),
        "gradient_relative_error": np.full((2, 4, 2), np.nan),
        "value_absolute_error": np.full((2, 4, 2), np.nan),
        "methods": ("naive", "corrected_B"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": json.dumps(sys.argv),
        "criterion": (
            "All fresh solves: active raw residual <= 1e-12. Each term: last two "
            "FD values agree within 1e-6 of max(abs(FD)) and both match exact AD "
            "at rtol=1e-6, atol=0. Both loose states: corrected gradient and value "
            "errors strictly decrease; naive gradient error > 1e-8 is resolvable. "
            "Each one-step polish strictly decreases the residual. Fresh public "
            "API DMerc parity: gradient norm difference <= 1e-10 relative; "
            "corrected value matches at rtol=1e-12, atol=1e-14."
        ),
    }


def measure(case: str, output: Path, max_steps: int) -> dict[str, Any]:
    """Checkpoint reference/FD successes before attempting nonlinear corrections."""
    data = initial_data(case)
    checkpoint(data, output, case)
    started = time.perf_counter()
    stage = "runtime"
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        enable_x64(jax.config)

        import vmecpp  # noqa: PLC0415
        from vmecpp import autodiff, autodiff_wout, geometry  # noqa: PLC0415
        from vmecpp._adjoint_contracts import corrected_vjp  # noqa: PLC0415
        from vmecpp.autodiff_corrections import (  # noqa: PLC0415
            geometry_objective_state_gradient,
        )
        from vmecpp.autodiff_diagnostics import raw_force  # noqa: PLC0415
        from vmecpp.autodiff_newton import newton_polish  # noqa: PLC0415
        from vmecpp.cpp import (  # noqa: PLC0415
            _vmecpp,  # pyright: ignore[reportAttributeAccessIssue]
        )

        sys.path.insert(0, str(ROOT / "tests"))
        from adjoint_verification.harness import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
            boundary_array,
            load_input,
            small_input,
        )

        binary = Path(_vmecpp.__file__).resolve()
        data["binary_path"] = str(binary)
        data["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        data["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        data["exact_force_jacobian"] = vmecpp.has_exact_force_jacobian()
        data["jax_version"] = jax.__version__
        data["jax_x64"] = jax.config.x64_enabled
        if not data["exact_force_jacobian"]:
            message = "Enzyme exact runtime unavailable; no numerical claim"
            raise RuntimeError(message)
        source = small_input() if case == "small" else load_input("li383_low_res")
        source.lbsubs = True
        source.always_fix_m1_gauge = True
        source.lforbal = False
        data["input_json"] = source.to_json()
        data["input_sha256"] = hashlib.sha256(data["input_json"].encode()).hexdigest()
        data["max_steps"] = max_steps
        boundary = boundary_array(source)
        direction = np.zeros_like(boundary)
        direction[0, 1, source.ntor + 1] = 1.0
        direction[0, 2, source.ntor + 1] = 0.3
        direction[1, 2, source.ntor] = -0.2
        direction /= np.linalg.norm(direction)
        data["boundary"] = boundary
        data["direction"] = direction
        checkpoint(data, output, case)

        def solve(parameters, label, solve_index):
            begin = time.perf_counter()
            model = autodiff._solve_model(source._to_cpp_vmecindata(), parameters)
            data[f"{label}_vmec_seconds"] = time.perf_counter() - begin
            data[f"{label}_vmec_residual"] = (model.fsqr, model.fsqz, model.fsql)
            data[f"{label}_initial_state"] = np.asarray(model.get_state()).copy()
            polish_begin = time.perf_counter()
            polish = newton_polish(model, max_steps, TOL)
            data[f"{label}_newton_seconds"] = time.perf_counter() - polish_begin
            data[f"{label}_residual_history"] = polish.residual_history
            data[f"{label}_state"] = polish.state
            data[f"{label}_active"] = polish.active
            data[f"{label}_seconds"] = time.perf_counter() - begin
            norm = float(np.linalg.norm(raw_force(model)[polish.active]))
            data[f"{label}_raw_residual"] = norm
            resolved = polish.converged and norm <= TOL
            data["solve_status"][solve_index] = (
                "passed" if resolved else "failed_residual"
            )
            checkpoint(data, output, case)
            if not resolved:
                message = f"{label}: active residual {norm:.6e} exceeds {TOL:.1e}"
                raise RuntimeError(message)
            return model, geometry.from_cpp(model.get_geometry())

        stage = "reference"
        reference_model, reference_geometry = solve(boundary, "reference", 0)

        def terms(value):
            fields = autodiff_wout.wout_quantities(value, source)
            selected = []
            for name in TERMS:
                field = fields[name]
                if field is None:
                    message = f"missing interior Mercier term {name}"
                    raise RuntimeError(message)
                selected.append(field)
            return jnp.stack(selected)[:, 1:-1]

        data["reference_mercier_terms"] = np.asarray(terms(reference_geometry))
        scales = np.max(np.abs(data["reference_mercier_terms"]), axis=1)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            message = "nonfinite or zero reference Mercier scale"
            raise RuntimeError(message)
        data["objective_scale"] = scales

        @jax.jit
        def objectives(value):
            normalized = terms(value) / jnp.asarray(scales)[:, None]
            weights = jnp.linspace(1.0, 2.0, normalized.shape[1])
            return jnp.mean(weights * normalized**2, axis=1)

        differentiated = jax.jit(
            jax.value_and_grad(lambda value, index: objectives(value)[index])
        )

        def cotangent(model, index, candidate=None):
            original = np.asarray(model.get_state()).copy()
            try:
                if candidate is not None:
                    model.set_state(np.ascontiguousarray(candidate))
                raw_force(model)
                value, bar = differentiated(
                    geometry.from_cpp(model.get_geometry()), index
                )
                coefficients = np.concatenate(
                    [
                        np.asarray(getattr(bar, name)).ravel()
                        for name in autodiff._GEOMETRY_COEFFICIENTS
                    ]
                )
                state_bar = np.asarray(
                    model.geometry_state_vjp(
                        np.ascontiguousarray(coefficients),
                        np.ascontiguousarray(bar.poloidal_flux),
                    )
                ).copy()
                flat_bar = np.concatenate(
                    [np.asarray(leaf).ravel() for leaf in jax.tree.leaves(bar)]
                )
                if not np.all(np.isfinite(state_bar)) or not np.isfinite(float(value)):
                    message = "nonfinite Mercier objective/cotangent"
                    raise RuntimeError(message)
                return float(value), state_bar, flat_bar
            finally:
                model.set_state(np.ascontiguousarray(original))
                raw_force(model)
                model.set_state(np.ascontiguousarray(original))

        def implicit(model, flat_bar):
            state = np.asarray(model.get_state()).copy()
            try:
                gradient, _ = autodiff._implicit_vjp(model, flat_bar)
                return np.asarray(gradient).copy()
            finally:
                model.set_state(np.ascontiguousarray(state))
                raw_force(model)
                model.set_state(np.ascontiguousarray(state))

        gradients, values = [], []
        stage = "reference_gradients"
        for index, term in enumerate(TERMS):
            begin = time.perf_counter()
            value, state_bar, flat_bar = cotangent(reference_model, index)
            gradients.append(implicit(reference_model, flat_bar))
            values.append(value)
            data[f"reference_{term}_state_cotangent"] = state_bar
            data[f"reference_{term}_seconds"] = time.perf_counter() - begin
            data["reference_gradient"] = np.asarray(gradients)
            data["reference_value"] = np.asarray(values)
            checkpoint(data, output, case)
        gradients = np.asarray(gradients)
        data["ad_directional"] = np.sum(gradients * direction[None], axis=(1, 2, 3))
        if not np.all(np.isfinite(data["ad_directional"])):
            message = "nonfinite exact directional gradient"
            raise RuntimeError(message)

        stage = "independent_fd"
        for step_index, step in enumerate(STEPS):
            for sign_index, sign in enumerate((1, -1)):
                label = f"{'plus' if sign == 1 else 'minus'}_{step_index}"
                _, value = solve(
                    boundary + sign * step * direction,
                    label,
                    1 + 2 * step_index + sign_index,
                )
                data["fd_values"][step_index, sign_index] = np.asarray(
                    objectives(value)
                )
                checkpoint(data, output, case)
            data["fd_derivative"][step_index] = (
                data["fd_values"][step_index, 0] - data["fd_values"][step_index, 1]
            ) / (2 * step)
            checkpoint(data, output, case)
        fd = data["fd_derivative"]
        magnitude = np.max(np.abs(fd), axis=0)
        valid = np.isfinite(magnitude) & (magnitude > 0)
        data["fd_relative_error"] = np.abs(fd - data["ad_directional"]) / np.where(
            valid, magnitude, np.nan
        )
        data["fd_plateau_relative_change"] = np.abs(fd[-1] - fd[-2]) / np.where(
            valid, magnitude, np.nan
        )
        for index in range(4):
            plateau = (
                valid[index] and data["fd_plateau_relative_change"][index] <= FD_RTOL
            )
            matched = np.all(
                np.isclose(
                    fd[-2:, index], data["ad_directional"][index], rtol=FD_RTOL, atol=0
                )
            )
            data["fd_status"][index] = (
                "passed"
                if plateau and matched
                else "failed_gradient"
                if plateau
                else "failed_plateau"
            )
        checkpoint(data, output, case)

        stage = "loose_solve"
        loose = source.model_copy(deep=True)
        loose.ftol_array = np.full_like(source.ftol_array, 1e-8)
        data["loose_input_json"] = loose.to_json()
        begin = time.perf_counter()
        model = autodiff._solve_model(loose._to_cpp_vmecindata(), boundary)
        data["loose_solve_seconds"] = time.perf_counter() - begin
        for state_index in range(2):
            data[f"loose_{state_index}_state"] = np.asarray(model.get_state()).copy()
            for index, term in enumerate(TERMS):
                stage = f"correction_{state_index}_{term}"
                begin = time.perf_counter()
                original_state = np.asarray(model.get_state()).copy()
                original_force = raw_force(model).copy()
                value, state_bar, flat_bar = cotangent(model, index)
                data[f"{stage}_objective_seconds"] = time.perf_counter() - begin
                naive_begin = time.perf_counter()
                naive = implicit(model, flat_bar)
                data[f"{stage}_naive_vjp_seconds"] = time.perf_counter() - naive_begin

                def scalar_objective(value, i=index):
                    return objectives(value)[i]

                callback = geometry_objective_state_gradient(model, scalar_objective)
                data["objective_state_gradient_implementation"] = (
                    "vmecpp.autodiff_corrections.geometry_objective_state_gradient"
                )

                corrected_begin = time.perf_counter()
                corrected, value_correction, diagnostics = corrected_vjp(
                    model,
                    state_bar,
                    "B",
                    objective_state_gradient=callback,
                    directional_step=1e-3,
                )
                data[f"{stage}_corrected_vjp_seconds"] = (
                    time.perf_counter() - corrected_begin
                )
                restored = np.array_equal(model.get_state(), original_state)
                force_restored = np.allclose(
                    raw_force(model), original_force, rtol=1e-12, atol=0
                )
                data[f"{stage}_state_restored"] = restored
                data[f"{stage}_force_restored"] = force_restored
                if not restored or not force_restored:
                    message = "nonlinear correction did not restore state/raw force"
                    raise RuntimeError(message)
                data[f"{stage}_seconds"] = time.perf_counter() - begin
                data[f"{stage}_diagnostics"] = json.dumps(
                    dataclasses.asdict(diagnostics)
                )
                data[f"{stage}_naive_gradient"] = naive
                data[f"{stage}_corrected_gradient"] = corrected
                data[f"{stage}_values"] = (value, value + value_correction)
                gradient_scale = np.linalg.norm(gradients[index])
                errors = (
                    np.asarray(
                        [
                            np.linalg.norm(naive - gradients[index]),
                            np.linalg.norm(corrected - gradients[index]),
                        ]
                    )
                    / gradient_scale
                )
                value_errors = np.abs(
                    np.asarray([value, value + value_correction]) - values[index]
                )
                data["correction_residual"][state_index, index] = (
                    diagnostics.force_residual
                )
                data["gradient_relative_error"][state_index, index] = errors
                data["value_absolute_error"][state_index, index] = value_errors
                data["correction_status"][state_index, index] = (
                    "unresolved_floor"
                    if errors[0] <= 1e-8
                    else "passed"
                    if np.all(np.isfinite(errors))
                    and errors[1] < errors[0]
                    and value_errors[1] < value_errors[0]
                    else "failed_improvement"
                )
                checkpoint(data, output, case)
            stage = f"loose_{state_index}_polish"
            begin = time.perf_counter()
            polish = newton_polish(model, 1, TOL)
            data[f"{stage}_seconds"] = time.perf_counter() - begin
            data[f"{stage}_residual_history"] = polish.residual_history
            decreasing = polish.residual_history[-1] < polish.residual_history[0]
            data[f"{stage}_status"] = "passed" if decreasing else "failed_decrease"
            checkpoint(data, output, case)
        stage = "public_api_DMerc"
        begin = time.perf_counter()

        def public_objective(value):
            return objectives(value)[0]

        public_value, public_gradient, public_diagnostics = autodiff.make_solver(
            loose
        ).corrected_value_and_grad(public_objective, boundary, directional_step=1e-3)
        data["public_api_seconds"] = time.perf_counter() - begin
        data["public_api_gradient"] = public_gradient
        data["public_api_value"] = public_value
        data["public_api_diagnostics"] = json.dumps(
            dataclasses.asdict(public_diagnostics)
        )
        direct_gradient = data["correction_0_DMerc_corrected_gradient"]
        direct_value = data["correction_0_DMerc_values"][1]
        parity_error = float(
            np.linalg.norm(public_gradient - direct_gradient)
            / max(float(np.linalg.norm(direct_gradient)), np.finfo(float).tiny)
        )
        data["public_api_gradient_relative_difference"] = parity_error
        data["public_api_value_absolute_difference"] = abs(public_value - direct_value)
        data["public_api_status"] = (
            "passed"
            if np.isfinite(parity_error)
            and parity_error <= 1e-10
            and np.isclose(public_value, direct_value, rtol=1e-12, atol=1e-14)
            else "failed_parity"
        )
        checkpoint(data, output, case)
        passed = (
            np.all(data["fd_status"] == "passed")
            and np.all(data["correction_status"] == "passed")
            and data["public_api_status"] == "passed"
            and all(
                data[f"loose_{index}_polish_status"] == "passed" for index in range(2)
            )
        )
        data["status"] = "passed" if passed else "failed_criteria"
    except Exception as error:
        data["status"] = "failed"
        data["failure"] = f"{stage}: {type(error).__name__}: {error}"
    finally:
        data["total_seconds"] = time.perf_counter() - started
        checkpoint(data, output, case)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=("small", "li383_low_res"), default=["small"]
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(RESULTS.resolve()) or args.max_steps < 1:
        parser.error("output must be under results and --max-steps must be positive")
    failed = False
    for case in args.cases:
        if args.render_only:
            with np.load(
                output / f"plot_08_mercier_corrections_{case}.npz", allow_pickle=False
            ) as saved:
                data = {name: saved[name] for name in saved.files}
        else:
            data = measure(case, output, args.max_steps)
        render(data, output, case)
        print(  # noqa: T201
            f"{case}: {data['status']}; FD={data['fd_status']}; "
            f"corrections={data['correction_status']}; "
            f"public API={data['public_api_status']}; failure={data['failure']}"
        )
        failed |= str(data["status"]) != "passed"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
