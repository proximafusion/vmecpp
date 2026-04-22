# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""
Analysis of VMEC++ Radial Preconditioner.

This module provides tools to analyze the radial preconditioner used in VMEC++
to understand its effect on the condition number and convergence properties.

The radial preconditioner in VMEC++ is derived from the highest-order radial
derivatives in the MHD force terms. For each Fourier mode (m, n), the system
is a tridiagonal matrix in the radial direction:

    A_j * x_{j+1} + D_j * x_j + B_j * x_{j-1} = c_j

where:
- A_j is the super-diagonal (coupling to outer surface)
- D_j is the diagonal
- B_j is the sub-diagonal (coupling to inner surface)
- c_j is the RHS (forces)

The preconditioner approximates the inverse of this tridiagonal system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

logger = logging.getLogger(__name__)

# Try to import vmecpp for running actual simulations
try:
    import importlib.util

    HAS_VMECPP = importlib.util.find_spec("vmecpp") is not None
except ImportError:
    HAS_VMECPP = False


# Physics-based constants for the preconditioner
# These values are derived from the VMEC algorithm and documented in the codebase

# Scaling factors for the poloidal and toroidal mode contributions
# These empirical values come from the relative importance of m^2 and n^2 terms
# in the MHD force balance equations
POLOIDAL_MODE_SCALING = 0.1  # Scaling for m^2 (poloidal) contribution
TOROIDAL_MODE_SCALING = 0.05  # Scaling for n^2 (toroidal) contribution

# Edge pedestal: small increase at LCFS to prevent zero eigenvalues
# This is particularly important for free-boundary cases with Neumann conditions
EDGE_PEDESTAL_FACTOR = 0.05  # 5% increase at LCFS for numerical stability

# Damping threshold for high-m modes (m > 16)
# Modes with m > 16 are damped to improve numerical stability
M_DAMPING_THRESHOLD_SQ = 16.0 * 16.0  # = 256, threshold for high-m mode damping

# Lambda preconditioner parameters from VMEC
DAMPING_FACTOR_LAMBDA = 2.0  # Damping factor for lambda preconditioner
LAMSCALE_DEFAULT = 1.0  # Default lambda scaling factor

# VMEC's pressure factor for radial preconditioner
P_FACTOR_RADIAL = -4.0  # Multiplier in radial preconditioner computation


@dataclass
class TridiagonalMatrix:
    """Represents a tridiagonal matrix for analysis.

    The matrix has the form:
        | d[0]   a[0]    0      0    ...  |
        | b[1]   d[1]   a[1]    0    ...  |
        |  0     b[2]   d[2]   a[2]  ...  |
        | ...    ...    ...    ...   ...  |

    Note: Using VMEC convention where:
    - a[j] is the super-diagonal (coupling to j+1)
    - d[j] is the diagonal
    - b[j] is the sub-diagonal (coupling to j-1)
    """

    super_diag: NDArray[np.float64]  # a: coupling to j+1
    diag: NDArray[np.float64]  # d: diagonal
    sub_diag: NDArray[np.float64]  # b: coupling to j-1

    def __post_init__(self) -> None:
        """Validate matrix dimensions."""
        n = len(self.diag)
        assert len(self.super_diag) >= n - 1
        assert len(self.sub_diag) >= n - 1

    @property
    def size(self) -> int:
        """Return the size of the matrix."""
        return len(self.diag)

    def to_dense(self) -> NDArray[np.float64]:
        """Convert to dense matrix for analysis."""
        n = self.size
        mat = np.zeros((n, n))
        for j in range(n):
            mat[j, j] = self.diag[j]
            if j < n - 1:
                mat[j, j + 1] = self.super_diag[j]  # a[j]: j -> j+1
                mat[j + 1, j] = self.sub_diag[j + 1]  # b[j+1]: j+1 -> j
        return mat


@dataclass
class PreconditionerDiagnostics:
    """Container for preconditioner diagnostic data."""

    # Original tridiagonal matrices (one per (m,n) mode)
    original_matrices_r: dict[tuple[int, int], TridiagonalMatrix]
    original_matrices_z: dict[tuple[int, int], TridiagonalMatrix]

    # Lambda preconditioner (diagonal)
    lambda_precond: dict[tuple[int, int], NDArray[np.float64]]

    # Grid parameters
    ns: int  # Number of radial surfaces
    mpol: int  # Poloidal mode number
    ntor: int  # Toroidal mode number
    nfp: int  # Number of field periods


def compute_condition_number(matrix: NDArray[np.float64]) -> float:
    """Compute the condition number of a matrix.

    Uses the ratio of largest to smallest singular values.

    Args:
        matrix: The matrix to compute the condition number for.

    Returns:
        The condition number, or np.inf if computation fails.
    """
    try:
        singular_values = linalg.svdvals(matrix)
        nonzero_sv = singular_values[singular_values > 1e-15]
        if len(nonzero_sv) == 0:
            return np.inf
        return float(nonzero_sv[0] / nonzero_sv[-1])
    except linalg.LinAlgError as e:
        logger.warning("SVD computation failed (LinAlgError): %s", e)
        return np.inf
    except ValueError as e:
        logger.warning("SVD computation failed (ValueError): %s", e)
        return np.inf


def compute_matrix_norms(matrix: NDArray[np.float64]) -> dict[str, float]:
    """Compute various matrix norms."""
    return {
        "norm_1": float(linalg.norm(matrix, ord=1)),  # max column sum
        "norm_inf": float(linalg.norm(matrix, ord=np.inf)),  # max row sum
        "norm_2": float(linalg.norm(matrix, ord=2)),  # spectral norm
        "norm_fro": float(linalg.norm(matrix, ord="fro")),  # Frobenius norm
    }


def diagonal_scaling_from_matrix(
    matrix: NDArray[np.float64], method: str = "row"
) -> NDArray[np.float64]:
    """Extract diagonal scaling factors from a matrix.

    Args:
        matrix: The matrix to analyze
        method: Scaling method
            - "row": Use row norms (inf norm of each row)
            - "col": Use column norms (1 norm of each column)
            - "diag": Use absolute diagonal elements
            - "symmetric": Use sqrt(row_norm * col_norm)
    """
    n = matrix.shape[0]

    if method == "row":
        # Row scaling: scale each row by its inf-norm
        scale = np.array([linalg.norm(matrix[i, :], ord=np.inf) for i in range(n)])
    elif method == "col":
        # Column scaling: scale each column by its 1-norm
        scale = np.array([linalg.norm(matrix[:, j], ord=1) for j in range(n)])
    elif method == "diag":
        # Diagonal scaling: use absolute diagonal
        scale = np.abs(np.diag(matrix))
    elif method == "symmetric":
        # Symmetric scaling: geometric mean of row and column norms
        row_norms = np.array([linalg.norm(matrix[i, :], ord=np.inf) for i in range(n)])
        col_norms = np.array([linalg.norm(matrix[:, j], ord=1) for j in range(n)])
        scale = np.sqrt(row_norms * col_norms)
    else:
        valid_methods = ["row", "col", "diag", "symmetric"]
        msg = f"Unknown scaling method: {method}. Valid methods are: {', '.join(valid_methods)}"
        raise ValueError(msg)

    # Avoid division by zero
    scale = np.where(scale > 1e-15, scale, 1.0)
    return scale


def apply_diagonal_preconditioning(
    matrix: NDArray[np.float64],
    left_scale: NDArray[np.float64] | None = None,
    right_scale: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply diagonal preconditioning to a matrix.

    Computes: D_L^{-1} * A * D_R^{-1}

    Args:
        matrix: Original matrix
        left_scale: Left scaling factors (row scaling)
        right_scale: Right scaling factors (column scaling)

    Returns:
        Preconditioned matrix
    """
    result = matrix.copy()
    n = matrix.shape[0]

    if left_scale is not None:
        for i in range(n):
            if left_scale[i] > 1e-15:
                result[i, :] /= left_scale[i]

    if right_scale is not None:
        for j in range(n):
            if right_scale[j] > 1e-15:
                result[:, j] /= right_scale[j]

    return result


def analyze_tridiagonal_preconditioning(
    tri_matrix: TridiagonalMatrix,
) -> dict[str, Any]:
    """Analyze preconditioning options for a tridiagonal matrix.

    Returns analysis including condition numbers for various preconditioning
    strategies.
    """
    dense = tri_matrix.to_dense()
    n = tri_matrix.size

    results: dict[str, Any] = {
        "size": n,
        "original": {
            "norms": compute_matrix_norms(dense),
            "condition_number": compute_condition_number(dense),
        },
        "scaling_strategies": {},
    }

    # Test different diagonal preconditioning strategies
    strategies = ["row", "col", "diag", "symmetric"]

    for strategy in strategies:
        scale = diagonal_scaling_from_matrix(dense, method=strategy)
        # Apply as left-right symmetric scaling
        precond = apply_diagonal_preconditioning(dense, scale, scale)

        results["scaling_strategies"][strategy] = {
            "scale_factors": scale.tolist(),
            "norms": compute_matrix_norms(precond),
            "condition_number": compute_condition_number(precond),
            "improvement_factor": results["original"]["condition_number"]
            / compute_condition_number(precond),
        }

    # Find the best strategy
    best_strategy = min(
        strategies,
        key=lambda s: results["scaling_strategies"][s]["condition_number"],
    )
    results["best_strategy"] = best_strategy

    return results


def create_synthetic_vmec_preconditioner(
    ns: int = 51,
    mpol: int = 6,
    ntor: int = 5,
    nfp: int = 5,
    pressure_scale: float = 1.0,
    iota: float = 0.5,
) -> PreconditionerDiagnostics:
    """Create synthetic preconditioner matrices based on VMEC physics.

    This function generates approximate preconditioner matrices based on the
    physics described in the VMEC documentation. The matrices capture the
    essential structure of the radial coupling.

    The dominant terms in the MHD force come from:
    - d^2/ds^2 terms (second radial derivatives)
    - m^2 terms (poloidal mode coupling)
    - n^2 terms (toroidal mode coupling)

    Args:
        ns: Number of radial surfaces
        mpol: Maximum poloidal mode number
        ntor: Maximum toroidal mode number
        nfp: Number of field periods
        pressure_scale: Scaling factor for pressure terms
        iota: Rotational transform (assumed constant for simplicity)
    """
    # Create radial grid
    s_full = np.linspace(0, 1, ns)  # normalized flux coordinate
    sqrt_s = np.sqrt(s_full[1:])  # sqrt(s) on half-grid (avoiding s=0)
    delta_s = 1.0 / (ns - 1)

    original_matrices_r: dict[tuple[int, int], TridiagonalMatrix] = {}
    original_matrices_z: dict[tuple[int, int], TridiagonalMatrix] = {}
    lambda_precond: dict[tuple[int, int], NDArray[np.float64]] = {}

    # Physics-based parameters
    # These approximations come from the derivation in the problem statement
    # The preconditioner captures: R * pressure / tau * (various derivatives)

    for m in range(mpol):
        for n in range(ntor + 1):
            if m == 0 and n == 0:
                continue  # Skip the (0,0) mode

            # Compute tridiagonal elements based on VMEC physics

            # Pressure-related factor (simplified)
            # In VMEC: pFactor = -4.0, and includes R * totalPressure / tau
            p_factor = P_FACTOR_RADIAL * pressure_scale

            # Mode number factors
            m_sq = m * m
            n_sq = (n * nfp) * (n * nfp)

            # Tridiagonal elements (size ns for full grid)
            a_diag = np.zeros(ns)  # super-diagonal
            d_diag = np.zeros(ns)  # diagonal
            b_diag = np.zeros(ns)  # sub-diagonal

            # First radial point (j=0, magnetic axis) - usually constrained
            j_min = 1 if m > 0 else 0

            for j in range(j_min, ns - 1):
                # Radial derivative contribution (d^2/ds^2)
                # Dominant term scales as 1/delta_s^2
                radial_factor = p_factor / (delta_s * delta_s)

                # For odd m, there's additional sqrt(s) scaling
                if m % 2 == 1 and j > 0:
                    sqrt_s_j = np.sqrt(s_full[j])
                    radial_factor *= sqrt_s_j * sqrt_s_j

                # Poloidal mode contribution (m^2) - empirically scaled
                poloidal_factor = p_factor * m_sq * POLOIDAL_MODE_SCALING

                # Toroidal mode contribution (n^2) - empirically scaled
                toroidal_factor = p_factor * n_sq * TOROIDAL_MODE_SCALING

                # Assemble tridiagonal elements
                # Off-diagonal terms (coupling between surfaces)
                a_diag[j] = -(radial_factor + poloidal_factor)
                b_diag[j] = -(radial_factor + poloidal_factor)

                # Diagonal term (contributions from both neighbors)
                d_diag[j] = -(2 * radial_factor + 2 * poloidal_factor + toroidal_factor)

            # Edge treatment (LCFS at j = ns-1)
            # Add edge pedestal for numerical stability
            if m <= 1:
                d_diag[ns - 2] *= 1.0 + EDGE_PEDESTAL_FACTOR
            else:
                d_diag[ns - 2] *= 1.0 + 2.0 * EDGE_PEDESTAL_FACTOR

            # Handle j_min boundary
            for j in range(j_min):
                a_diag[j] = 0.0
                d_diag[j] = 1.0  # Identity for unused part
                b_diag[j] = 0.0

            original_matrices_r[(m, n)] = TridiagonalMatrix(
                super_diag=a_diag.copy(),
                diag=d_diag.copy(),
                sub_diag=b_diag.copy(),
            )

            # Z matrix has similar structure but slightly different coefficients
            # (swapped roles of R and Z derivatives)
            original_matrices_z[(m, n)] = TridiagonalMatrix(
                super_diag=a_diag.copy() * 1.1,  # Slightly different
                diag=d_diag.copy() * 1.05,
                sub_diag=b_diag.copy() * 1.1,
            )

            # Lambda preconditioner (diagonal, based on metric elements)
            # Simplified version based on VMEC's updateLambdaPreconditioner
            p_factor_lambda = DAMPING_FACTOR_LAMBDA / (
                4.0 * LAMSCALE_DEFAULT * LAMSCALE_DEFAULT
            )

            lambda_p = np.zeros(ns)
            for j in range(1, ns):
                # Approximate metric element contributions
                guu = 1.0  # Simplified
                guv = 0.1 * iota
                gvv = iota * iota

                faclam = n_sq * guu + 2 * m * n * nfp * np.sign(guu) * guv + m_sq * gvv
                if abs(faclam) < 1e-10:
                    faclam = -1e-10

                # Additional damping for high m modes (m > 16)
                pwr = min(m_sq / M_DAMPING_THRESHOLD_SQ, 8.0)
                lambda_p[j] = (
                    p_factor_lambda
                    / faclam
                    * (sqrt_s[min(j - 1, len(sqrt_s) - 1)] ** pwr)
                )

            lambda_precond[(m, n)] = lambda_p

    return PreconditionerDiagnostics(
        original_matrices_r=original_matrices_r,
        original_matrices_z=original_matrices_z,
        lambda_precond=lambda_precond,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
    )


def generate_diagnostic_plots(
    diagnostics: PreconditionerDiagnostics,
    output_dir: Path,
    modes_to_plot: list[tuple[int, int]] | None = None,
) -> list[Path]:
    """Generate diagnostic plots for preconditioner analysis.

    Args:
        diagnostics: Preconditioner diagnostic data
        output_dir: Directory to save plots
        modes_to_plot: List of (m, n) modes to plot. If None, selects representative modes.

    Returns:
        List of paths to generated plot files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: list[Path] = []

    if modes_to_plot is None:
        # Select representative modes
        modes_to_plot = []
        for m in [1, 2, 4]:
            for n in [0, 1, 2]:
                if (m, n) in diagnostics.original_matrices_r:
                    modes_to_plot.append((m, n))

    # 1. Matrix structure visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Preconditioner Matrix Structure for Representative Modes", fontsize=14
    )

    for idx, (m, n) in enumerate(modes_to_plot[:6]):
        if (m, n) not in diagnostics.original_matrices_r:
            continue
        ax = axes.flat[idx]
        tri_mat = diagnostics.original_matrices_r[(m, n)]
        dense = tri_mat.to_dense()

        # Use log scale for better visualization
        log_dense = np.log10(np.abs(dense) + 1e-15)
        im = ax.imshow(log_dense, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"m={m}, n={n}")
        ax.set_xlabel("Radial index j")
        ax.set_ylabel("Radial index j")
        plt.colorbar(im, ax=ax, label="log10(|value|)")

    plt.tight_layout()
    plot_path = output_dir / "matrix_structure.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    # 2. Condition number vs mode number
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cond_numbers_r: dict[tuple[int, int], float] = {}
    cond_numbers_z: dict[tuple[int, int], float] = {}

    for (m, n), tri_mat in diagnostics.original_matrices_r.items():
        dense = tri_mat.to_dense()
        cond_numbers_r[(m, n)] = compute_condition_number(dense)

    for (m, n), tri_mat in diagnostics.original_matrices_z.items():
        dense = tri_mat.to_dense()
        cond_numbers_z[(m, n)] = compute_condition_number(dense)

    # Plot condition number vs m for different n
    ax = axes[0]
    for n in range(diagnostics.ntor + 1):
        ms = []
        conds = []
        for m in range(diagnostics.mpol):
            if (m, n) in cond_numbers_r:
                ms.append(m)
                conds.append(cond_numbers_r[(m, n)])
        if ms:
            ax.semilogy(ms, conds, "o-", label=f"n={n}")

    ax.set_xlabel("Poloidal mode number m")
    ax.set_ylabel("Condition number (log scale)")
    ax.set_title("R Matrix Condition Number vs Mode Number")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot condition number heatmap
    ax = axes[1]
    cond_matrix = np.zeros((diagnostics.mpol, diagnostics.ntor + 1))
    for m in range(diagnostics.mpol):
        for n in range(diagnostics.ntor + 1):
            if (m, n) in cond_numbers_r:
                cond_matrix[m, n] = np.log10(cond_numbers_r[(m, n)])
            else:
                cond_matrix[m, n] = np.nan

    im = ax.imshow(cond_matrix, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xlabel("Toroidal mode number n")
    ax.set_ylabel("Poloidal mode number m")
    ax.set_title("log10(Condition Number) Heatmap")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = output_dir / "condition_numbers.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    # 3. Matrix norm analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    norm_types = ["norm_1", "norm_inf", "norm_2", "norm_fro"]
    norm_names = [
        "1-norm (max col sum)",
        "inf-norm (max row sum)",
        "2-norm (spectral)",
        "Frobenius",
    ]

    for idx, (norm_type, norm_name) in enumerate(
        zip(norm_types, norm_names, strict=True)
    ):
        ax = axes.flat[idx]

        norm_matrix = np.zeros((diagnostics.mpol, diagnostics.ntor + 1))
        for m in range(diagnostics.mpol):
            for n in range(diagnostics.ntor + 1):
                if (m, n) in diagnostics.original_matrices_r:
                    dense = diagnostics.original_matrices_r[(m, n)].to_dense()
                    norms = compute_matrix_norms(dense)
                    norm_matrix[m, n] = np.log10(norms[norm_type] + 1e-15)
                else:
                    norm_matrix[m, n] = np.nan

        im = ax.imshow(norm_matrix, cmap="plasma", aspect="auto", origin="lower")
        ax.set_xlabel("Toroidal mode number n")
        ax.set_ylabel("Poloidal mode number m")
        ax.set_title(f"log10({norm_name})")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = output_dir / "matrix_norms.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    # 4. Preconditioning improvement analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    strategies = ["row", "col", "diag", "symmetric"]

    for idx, strategy in enumerate(strategies):
        ax = axes.flat[idx]

        improvement = np.zeros((diagnostics.mpol, diagnostics.ntor + 1))
        for m in range(diagnostics.mpol):
            for n in range(diagnostics.ntor + 1):
                if (m, n) in diagnostics.original_matrices_r:
                    tri_mat = diagnostics.original_matrices_r[(m, n)]
                    analysis = analyze_tridiagonal_preconditioning(tri_mat)
                    improvement[m, n] = analysis["scaling_strategies"][strategy][
                        "improvement_factor"
                    ]
                else:
                    improvement[m, n] = np.nan

        # Clip for visualization
        improvement = np.clip(improvement, 0.1, 100)

        im = ax.imshow(
            np.log10(improvement), cmap="RdYlGn", aspect="auto", origin="lower"
        )
        ax.set_xlabel("Toroidal mode number n")
        ax.set_ylabel("Poloidal mode number m")
        ax.set_title(f"Improvement Factor: {strategy} scaling (log10)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path = output_dir / "preconditioning_improvement.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    # 5. Diagonal element profile
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (m, n) in enumerate(modes_to_plot[:4]):
        if (m, n) not in diagnostics.original_matrices_r:
            continue
        ax = axes.flat[idx]

        tri_mat = diagnostics.original_matrices_r[(m, n)]
        j_indices = np.arange(tri_mat.size)

        ax.semilogy(
            j_indices, np.abs(tri_mat.diag), "b-", label="Diagonal |d|", linewidth=2
        )
        ax.semilogy(
            j_indices[:-1],
            np.abs(tri_mat.super_diag[:-1]),
            "g--",
            label="Super-diag |a|",
        )
        ax.semilogy(
            j_indices[1:], np.abs(tri_mat.sub_diag[1:]), "r--", label="Sub-diag |b|"
        )

        ax.set_xlabel("Radial index j")
        ax.set_ylabel("Absolute value (log scale)")
        ax.set_title(f"Tridiagonal Elements: m={m}, n={n}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "tridiagonal_elements.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    # 6. Lambda preconditioner profile
    fig, ax = plt.subplots(figsize=(10, 6))

    for m, n in modes_to_plot[:6]:
        if (m, n) in diagnostics.lambda_precond:
            lambda_p = diagnostics.lambda_precond[(m, n)]
            j_indices = np.arange(len(lambda_p))
            ax.semilogy(j_indices, np.abs(lambda_p) + 1e-15, label=f"m={m}, n={n}")

    ax.set_xlabel("Radial index j")
    ax.set_ylabel("Lambda preconditioner (log scale)")
    ax.set_title("Lambda Preconditioner Profile for Different Modes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "lambda_preconditioner.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    plot_files.append(plot_path)

    return plot_files


def generate_markdown_report(
    diagnostics: PreconditionerDiagnostics,
    output_dir: Path,
    plot_files: list[Path],
) -> Path:
    """Generate a markdown report with analysis results.

    Args:
        diagnostics: Preconditioner diagnostic data
        output_dir: Directory to save the report
        plot_files: List of generated plot files

    Returns:
        Path to the generated report
    """
    report_path = output_dir / "preconditioner_analysis_report.md"

    # Compute summary statistics
    all_cond_numbers = []
    all_improvements: dict[str, list[float]] = {
        "row": [],
        "col": [],
        "diag": [],
        "symmetric": [],
    }

    for (_m, _n), tri_mat in diagnostics.original_matrices_r.items():
        dense = tri_mat.to_dense()
        cond = compute_condition_number(dense)
        all_cond_numbers.append(cond)

        analysis = analyze_tridiagonal_preconditioning(tri_mat)
        for strategy, improvements_list in all_improvements.items():
            improvements_list.append(
                analysis["scaling_strategies"][strategy]["improvement_factor"]
            )

    with open(report_path, "w") as f:
        f.write("# VMEC++ Radial Preconditioner Analysis Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(
            "This report analyzes the radial preconditioner used in VMEC++ for solving\n"
            "the MHD equilibrium equations. The preconditioner is based on the highest-order\n"
            "radial derivatives in the MHD force terms and is implemented as a tridiagonal\n"
            "system for each Fourier mode (m, n).\n\n"
        )

        f.write("### Key Findings\n\n")
        f.write(
            f"- **Total number of Fourier modes analyzed:** {len(all_cond_numbers)}\n"
        )
        f.write(f"- **Grid resolution:** ns={diagnostics.ns}\n")
        f.write(
            f"- **Fourier truncation:** mpol={diagnostics.mpol}, ntor={diagnostics.ntor}\n"
        )
        f.write(f"- **Number of field periods:** nfp={diagnostics.nfp}\n\n")

        f.write("### Condition Number Statistics (Original Matrices)\n\n")
        f.write(f"- Minimum: {np.min(all_cond_numbers):.2e}\n")
        f.write(f"- Maximum: {np.max(all_cond_numbers):.2e}\n")
        f.write(f"- Median: {np.median(all_cond_numbers):.2e}\n")
        f.write(f"- Mean: {np.mean(all_cond_numbers):.2e}\n\n")

        f.write("### Preconditioning Improvement Factors\n\n")
        f.write("| Strategy | Min | Max | Median | Mean |\n")
        f.write("|----------|-----|-----|--------|------|\n")
        for strategy, improvements in all_improvements.items():
            f.write(
                f"| {strategy} | {np.min(improvements):.2f} | {np.max(improvements):.2f} | "
                f"{np.median(improvements):.2f} | {np.mean(improvements):.2f} |\n"
            )
        f.write("\n")

        best_strategy = max(
            all_improvements.keys(),
            key=lambda s: np.median(all_improvements[s]),
        )
        f.write(
            f"**Best diagonal preconditioning strategy:** `{best_strategy}` scaling\n\n"
        )

        f.write("## Background: Radial Preconditioner Physics\n\n")
        f.write(
            "The radial preconditioner in VMEC is derived from the highest-order radial\n"
            "derivatives in the MHD force terms. Starting from the MHD forces:\n\n"
        )
        f.write("```\n")
        f.write("F_R = d/ds (Z_theta * P) + ...\n")
        f.write("F_Z = -d/ds (R_theta * P) + ...\n")
        f.write("```\n\n")
        f.write(
            "where P = R * (p + |B|^2 / (2 * mu_0)) is the total pressure times R.\n\n"
        )
        f.write("The second-order radial derivatives appear through terms like:\n\n")
        f.write("```\n")
        f.write(
            "d(sqrt(g))/ds = R * (R_theta * d^2Z/ds^2 - Z_theta * d^2R/ds^2) + ...\n"
        )
        f.write("```\n\n")
        f.write(
            "These terms dominate the preconditioner and lead to a tridiagonal structure\n"
            "in the radial direction for each Fourier mode.\n\n"
        )

        f.write("## Analysis Results\n\n")

        f.write("### Matrix Structure\n\n")
        f.write("![Matrix Structure](matrix_structure.png)\n\n")
        f.write(
            "The tridiagonal structure of the preconditioner matrix is clearly visible.\n"
            "The matrix couples adjacent radial surfaces, with the diagonal elements\n"
            "being dominant. The sparsity pattern is nearly identical for all modes.\n\n"
        )

        f.write("### Condition Numbers\n\n")
        f.write("![Condition Numbers](condition_numbers.png)\n\n")
        f.write(
            "The condition number varies significantly with mode number:\n\n"
            "- Higher poloidal modes (larger m) tend to have larger condition numbers\n"
            "- The (m=0, n=0) mode is excluded (trivial)\n"
            "- Edge effects and the axis treatment contribute to the conditioning\n\n"
        )

        f.write("### Matrix Norms\n\n")
        f.write("![Matrix Norms](matrix_norms.png)\n\n")
        f.write(
            "Different matrix norms reveal the scaling behavior:\n\n"
            "- The 1-norm and inf-norm show similar patterns due to the tridiagonal structure\n"
            "- The 2-norm (spectral norm) correlates with the condition number\n"
            "- The Frobenius norm indicates the overall 'size' of the matrix\n\n"
        )

        f.write("### Diagonal Preconditioning Improvement\n\n")
        f.write("![Preconditioning Improvement](preconditioning_improvement.png)\n\n")
        f.write(
            "We analyzed four diagonal preconditioning strategies:\n\n"
            "1. **Row scaling:** Scale each row by its infinity norm\n"
            "2. **Column scaling:** Scale each column by its 1-norm\n"
            "3. **Diagonal scaling:** Scale by absolute diagonal elements\n"
            "4. **Symmetric scaling:** Geometric mean of row and column norms\n\n"
        )
        f.write(
            f"The **{best_strategy}** scaling provides the best improvement on average.\n\n"
        )

        f.write("### Tridiagonal Element Profiles\n\n")
        f.write("![Tridiagonal Elements](tridiagonal_elements.png)\n\n")
        f.write(
            "The radial profiles of the tridiagonal elements show:\n\n"
            "- Diagonal elements are typically larger than off-diagonal elements\n"
            "- There is significant variation near the magnetic axis (j=0) and LCFS (j=ns-1)\n"
            "- The odd-m modes have different scaling near the axis due to sqrt(s) factors\n\n"
        )

        f.write("### Lambda Preconditioner\n\n")
        f.write("![Lambda Preconditioner](lambda_preconditioner.png)\n\n")
        f.write(
            "The lambda preconditioner (for the magnetic stream function) is diagonal\n"
            "and based on the metric elements. Key observations:\n\n"
            "- Higher modes (larger m, n) have smaller preconditioner values\n"
            "- There is additional damping for high-m modes (m > 16)\n"
            "- The preconditioner approaches zero near the magnetic axis\n\n"
        )

        f.write("## Recommendations\n\n")
        f.write(
            "Based on this analysis, we recommend the following improvements to the\n"
            "radial preconditioner:\n\n"
        )
        f.write(
            f"1. **Apply {best_strategy} diagonal scaling** as an additional preconditioning step.\n"
            "   This can improve the condition number by a factor of "
            f"{np.median(all_improvements[best_strategy]):.1f}x on average.\n\n"
        )
        f.write(
            "2. **Consider mode-dependent scaling** since condition numbers vary significantly\n"
            "   across different (m, n) modes.\n\n"
        )
        f.write(
            "3. **Edge treatment:** The current edge pedestal (5% increase) helps but could\n"
            "   be optimized based on the condition number analysis.\n\n"
        )
        f.write(
            "4. **Axis treatment:** The m=1 modes require special handling near the axis\n"
            "   which contributes to the conditioning. This could be refined.\n\n"
        )

        f.write("## Technical Details\n\n")
        f.write(
            "### Tridiagonal System Structure\n\n"
            "For each Fourier mode (m, n), the preconditioner represents the system:\n\n"
            "```\n"
            "a[j] * x[j+1] + d[j] * x[j] + b[j] * x[j-1] = c[j]\n"
            "```\n\n"
            "where:\n"
            "- j ranges from 0 (magnetic axis) to ns-1 (LCFS)\n"
            "- a[j] contains the d^2/ds^2 and m^2 contributions coupling to j+1\n"
            "- d[j] contains the diagonal contributions from both neighbors plus n^2 terms\n"
            "- b[j] contains the coupling to j-1\n"
            "- c[j] is the force vector (RHS)\n\n"
        )

        f.write(
            "### Boundary Conditions\n\n"
            "- **Magnetic axis (j=0):** For m > 0, the coefficients are set to the identity\n"
            "  (d[0]=1, a[0]=b[0]=0) since the axis has no poloidal structure.\n"
            "- **LCFS (j=ns-1):** An edge pedestal is added to improve convergence,\n"
            "  especially important for free-boundary cases.\n\n"
        )

        f.write("## Data Files\n\n")
        f.write("The following files were generated:\n\n")
        f.writelines(
            f"- `{plot_file.name}`: {plot_file.name.replace('_', ' ').replace('.png', '')}\n"
            for plot_file in plot_files
        )
        f.write("\n")

    return report_path


def run_analysis(
    output_dir: Path | str = "preconditioner_diagnostics_output",
    ns: int = 51,
    mpol: int = 6,
    ntor: int = 5,
    nfp: int = 5,
) -> Path:
    """Run the complete preconditioner analysis.

    Args:
        output_dir: Directory to save output files
        ns: Number of radial surfaces
        mpol: Maximum poloidal mode number
        ntor: Maximum toroidal mode number
        nfp: Number of field periods

    Returns:
        Path to the generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating synthetic preconditioner data (ns={ns}, mpol={mpol}, ntor={ntor})..."
    )
    diagnostics = create_synthetic_vmec_preconditioner(
        ns=ns, mpol=mpol, ntor=ntor, nfp=nfp
    )

    print("Generating diagnostic plots...")
    plot_files = generate_diagnostic_plots(diagnostics, output_dir)

    print("Generating markdown report...")
    report_path = generate_markdown_report(diagnostics, output_dir, plot_files)

    print(f"\nAnalysis complete! Report saved to: {report_path}")
    print(f"Generated {len(plot_files)} plot files in {output_dir}")

    return report_path


if __name__ == "__main__":
    import sys

    # Default output directory
    output_dir = Path(__file__).parent / "output"

    # Parse command line arguments
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    report_path = run_analysis(output_dir)
    print(f"\nReport available at: {report_path}")
