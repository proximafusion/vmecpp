# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A content-addressed store of converged equilibria.

Design loops solve the same equilibrium, or a near neighbor of one, over and
over. :class:`EquilibriumCache` makes the repeated solve cheap: every converged
:class:`VmecOutput` is stored under a hash of the input that produced it, an
exact repeat is answered from the store, and a miss close to a stored entry
hot-restarts from that neighbor instead of iterating from scratch.

The key is content-addressed. The input is canonicalized by a round trip
through :class:`VmecInput` and, for a free-boundary case, the ``mgrid_file``
path is replaced by a hash of the file's content, so the key follows the
physics of the input rather than the file layout on disk.

A repeated solve is answered in one of two ways. With ``verify=True`` (the
default) the stored state is handed to :func:`vmecpp.run` as ``restart_from``,
so VMEC++ itself re-establishes force balance from it; starting from a
converged state this takes a couple of iterations instead of the full count,
and the returned output is the freshly converged one. With ``verify=False``
the stored output is returned as-is.

A miss consults the store for the nearest compatible neighbor, interpolates
its solution to the requested resolution with
:func:`vmecpp.interpolate_solution`, and hot-restarts from it; if that fails
for any reason, the solve falls back to a cold start. Hot-restarted solves run
the final ``ns_array`` entry only, like every hot restart in VMEC++.

:meth:`EquilibriumCache.submit` runs the same logic on a background thread and
returns a :class:`concurrent.futures.Future`; the C++ solver releases the GIL,
so submitted solves overlap with the caller and with each other up to
``max_parallel_solves``.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import gzip
import hashlib
import os
import tempfile
import threading
import typing
from pathlib import Path

import numpy as np

if typing.TYPE_CHECKING:
    from vmecpp import VmecInput, VmecOutput
    from vmecpp._free_boundary import MagneticFieldResponseTable

# Arrays compared by the nearest-neighbor distance, beyond the boundary and
# axis geometry. Lengths may differ between entries; shorter arrays are
# zero-padded.
_PROFILE_ARRAYS = (
    "am",
    "ai",
    "ac",
    "am_aux_s",
    "am_aux_f",
    "ai_aux_s",
    "ai_aux_f",
    "ac_aux_s",
    "ac_aux_f",
)

_SCALARS = ("phiedge", "curtor", "pres_scale", "gamma")


def _canonical_input(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None,
) -> VmecInput:
    """The input as hashed: revalidated, with ``mgrid_file`` content-addressed.

    For a free-boundary input the ``mgrid_file`` path is replaced by
    ``sha256:<hash>`` of the vacuum field actually used, so that moving or
    renaming the file does not change the key, while changing its content
    does. The caller's input is not modified.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    canonical = vmecpp.VmecInput.model_validate(input).model_copy(deep=True)
    if canonical.lfreeb:
        if magnetic_field is not None:
            digest = hashlib.sha256(
                magnetic_field.model_dump_json().encode()
            ).hexdigest()
        else:
            digest = hashlib.sha256(Path(canonical.mgrid_file).read_bytes()).hexdigest()
        canonical.mgrid_file = f"sha256:{digest}"
    return canonical


def cache_key(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None = None,
) -> str:
    """The content hash under which this input's equilibrium is stored.

    Two inputs get the same key exactly when their canonical serializations agree; every
    field that reaches the solver participates. For a free-boundary input the vacuum
    field content is hashed in, so the key is independent of where the mgrid file lives.
    """
    canonical = _canonical_input(input, magnetic_field)
    return hashlib.sha256(canonical.model_dump_json().encode()).hexdigest()


def _final_step_input(input: VmecInput) -> VmecInput:
    """The input reduced to its final multigrid step.

    Hot restart requires the restart state's resolution to match the first
    solved step, so a hot-restarted solve runs the last ``ns_array`` entry
    only, exactly as :func:`vmecpp.run` does between continuation steps. A
    sequence-valued ``mpol``/``ntor`` likewise collapses to its final entry.
    """
    update: dict[str, typing.Any] = {
        "ns_array": np.asarray(input.ns_array[-1:]),
        "ftol_array": np.asarray(input.ftol_array[-1:]),
        "niter_array": np.asarray(input.niter_array[-1:]),
    }
    if not isinstance(input.mpol, int):
        update["mpol"] = int(np.asarray(input.mpol)[-1])
    if not isinstance(input.ntor, int):
        update["ntor"] = int(np.asarray(input.ntor)[-1])
    return input.model_copy(update=update)


def _pad_modes(coefficients: np.ndarray, mpol: int, ntor: int) -> np.ndarray:
    """Zero-pad a ``[mpol, 2*ntor+1]`` coefficient array to a larger grid."""
    out = np.zeros((mpol, 2 * ntor + 1))
    rows, cols = coefficients.shape
    ntor_have = (cols - 1) // 2
    out[:rows, ntor - ntor_have : ntor + ntor_have + 1] = coefficients
    return out


def _block_distance(a: np.ndarray, b: np.ndarray) -> float:
    """A relative L2 distance between two arrays of equal shape."""
    return float(np.linalg.norm(a - b) / (1.0 + np.linalg.norm(b)))


def _input_distance(a: VmecInput, b: VmecInput) -> float:
    """How far apart two compatible inputs are, for picking a restart neighbor.

    A heuristic, not a metric with physical meaning: the boundary and axis Fourier
    coefficients dominate, with the profile arrays and the driving scalars contributing
    alongside. Arrays of different resolution are zero-padded to a common grid. Smaller
    is closer; identical inputs give 0.
    """
    distance = 0.0

    mpol = max(a.rbc.shape[0], b.rbc.shape[0])
    ntor = max((a.rbc.shape[1] - 1) // 2, (b.rbc.shape[1] - 1) // 2)
    for name in ("rbc", "zbs", "rbs", "zbc"):
        va = getattr(a, name, None)
        vb = getattr(b, name, None)
        if va is None and vb is None:
            continue
        pa = (
            _pad_modes(va, mpol, ntor)
            if va is not None
            else np.zeros((mpol, 2 * ntor + 1))
        )
        pb = (
            _pad_modes(vb, mpol, ntor)
            if vb is not None
            else np.zeros((mpol, 2 * ntor + 1))
        )
        distance += _block_distance(pa, pb)

    for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        va = getattr(a, name, None)
        vb = getattr(b, name, None)
        if va is None and vb is None:
            continue
        n = max(va.size if va is not None else 0, vb.size if vb is not None else 0)
        pa = np.zeros(n)
        pb = np.zeros(n)
        if va is not None:
            pa[: va.size] = va
        if vb is not None:
            pb[: vb.size] = vb
        distance += _block_distance(pa, pb)

    for name in _PROFILE_ARRAYS:
        va = np.asarray(getattr(a, name))
        vb = np.asarray(getattr(b, name))
        n = max(va.size, vb.size)
        pa = np.zeros(n)
        pb = np.zeros(n)
        pa[: va.size] = va
        pb[: vb.size] = vb
        distance += _block_distance(pa, pb)

    for name in _SCALARS:
        distance += abs(getattr(a, name) - getattr(b, name)) / (
            1.0 + abs(getattr(b, name))
        )

    return distance


def _compatible(a: VmecInput, b: VmecInput) -> bool:
    """Whether a stored entry can serve as a restart neighbor for this input."""
    if a.lasym != b.lasym or a.nfp != b.nfp or a.lfreeb != b.lfreeb:
        return False
    # For free-boundary entries the canonical mgrid_file carries the content
    # hash of the vacuum field, which must agree for the state to be reusable.
    return not a.lfreeb or a.mgrid_file == b.mgrid_file


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        Path(tmp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp).unlink()
        raise


class EquilibriumCache:
    """Solve equilibria through a persistent content-addressed store.

    Example:
        >>> import tempfile
        >>> import vmecpp
        >>> vmec_input = vmecpp.VmecInput.from_file("examples/data/solovev.json")
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     cache = vmecpp.EquilibriumCache(tmp)
        ...     cold = cache.solve(vmec_input, verbose=False, max_threads=1)
        ...     again = cache.solve(vmec_input, verbose=False, max_threads=1)
        >>> again.wout.niter < cold.wout.niter
        True

    Entries live under ``<path>/entries`` as one gzipped JSON per equilibrium
    plus a small sidecar carrying the canonical input, which is what the
    nearest-neighbor search reads. The directory can be shared between
    processes and machines; writes are atomic, and with ``verify=True`` a hit
    is never trusted blindly, since VMEC++ re-establishes force balance from
    the stored state before returning it.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        warm_start: bool = True,
        max_parallel_solves: int = 1,
    ) -> None:
        """Open (creating if needed) a cache rooted at ``path``.

        Args:
            path: directory holding the store; created if absent.
            warm_start: on a miss, hot-restart from the nearest compatible
                stored neighbor instead of solving from scratch. This pays off
                when the store holds a close neighbor; restarting from a
                distant state can take more iterations than the cold multigrid
                ramp it replaces.
            max_parallel_solves: how many :meth:`submit` solves may run
                concurrently. Each solve is itself OpenMP-parallel, so the
                default avoids oversubscription.
        """
        self.path = Path(path)
        self.warm_start = warm_start
        self._entries_dir = self.path / "entries"
        self._entries_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._max_parallel_solves = max_parallel_solves
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        # key -> canonical VmecInput, fed by the sidecar files.
        self._index: dict[str, VmecInput] = {}
        self._scan()

    # ------------------------------------------------------------------ store

    def _entry_path(self, key: str) -> Path:
        return self._entries_dir / f"{key}.json.gz"

    def _sidecar_path(self, key: str) -> Path:
        return self._entries_dir / f"{key}.input.json"

    def _scan(self) -> None:
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        for sidecar in self._entries_dir.glob("*.input.json"):
            key = sidecar.name[: -len(".input.json")]
            if key in self._index or not self._entry_path(key).exists():
                continue
            self._index[key] = vmecpp.VmecInput.model_validate_json(sidecar.read_text())

    def _store(self, key: str, output: VmecOutput, canonical: VmecInput) -> None:
        _atomic_write(
            self._entry_path(key), gzip.compress(output.model_dump_json().encode())
        )
        _atomic_write(self._sidecar_path(key), canonical.model_dump_json().encode())
        with self._lock:
            self._index[key] = canonical

    def _load(self, key: str) -> VmecOutput | None:
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        path = self._entry_path(key)
        if not path.exists():
            return None
        return vmecpp.VmecOutput.model_validate_json(gzip.decompress(path.read_bytes()))

    def keys(self) -> list[str]:
        """The keys of every stored equilibrium."""
        with self._lock:
            return sorted(self._index)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._index

    # ------------------------------------------------------------------ solve

    def _nearest_neighbor(self, canonical: VmecInput) -> VmecOutput | None:
        with self._lock:
            candidates = [
                (key, stored_input)
                for key, stored_input in self._index.items()
                if _compatible(stored_input, canonical)
            ]
        if not candidates:
            return None
        best_key = min(
            candidates, key=lambda item: _input_distance(item[1], canonical)
        )[0]
        return self._load(best_key)

    def solve(
        self,
        input: VmecInput,
        magnetic_field: MagneticFieldResponseTable | None = None,
        *,
        verify: bool = True,
        max_threads: int | None = None,
        verbose: bool | int | None = None,
    ) -> VmecOutput:
        """:func:`vmecpp.run` through the store.

        An exact repeat of a stored input is answered from the store: with
        ``verify=True`` by hot-restarting VMEC++ from the stored state, which
        re-establishes force balance in a couple of iterations and returns the
        fresh output; with ``verify=False`` by returning the stored output
        directly. A miss is solved and stored, hot-restarting from the nearest
        compatible stored neighbor when ``warm_start`` is on. Hot-restarted
        solves run the final ``ns_array`` entry only, so their ``output.input``
        carries the collapsed schedule, as with any hot restart.

        Args:
            input: as for :func:`vmecpp.run`.
            magnetic_field: as for :func:`vmecpp.run`.
            verify: re-establish force balance on a hit instead of trusting
                the stored bytes.
            max_threads: as for :func:`vmecpp.run`.
            verbose: as for :func:`vmecpp.run`; ``None`` keeps its default.
        """
        import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

        run_kwargs: dict[str, typing.Any] = {"max_threads": max_threads}
        if verbose is not None:
            run_kwargs["verbose"] = verbose

        canonical = _canonical_input(input, magnetic_field)
        key = hashlib.sha256(canonical.model_dump_json().encode()).hexdigest()

        stored = self._load(key)
        if stored is not None:
            if not verify:
                return stored
            try:
                return vmecpp.run(
                    _final_step_input(vmecpp.VmecInput.model_validate(input)),
                    magnetic_field,
                    restart_from=stored,
                    **run_kwargs,
                )
            except Exception:
                # The stored state failed to re-converge; fall through to a
                # cold solve, which overwrites the entry.
                pass

        if stored is None and self.warm_start:
            neighbor = self._nearest_neighbor(canonical)
            if neighbor is not None:
                try:
                    final = _final_step_input(vmecpp.VmecInput.model_validate(input))
                    interpolated = vmecpp.interpolate_solution(neighbor, final)
                    output = vmecpp.run(
                        final, magnetic_field, restart_from=interpolated, **run_kwargs
                    )
                except Exception:
                    # A neighbor is a guess; whatever went wrong, the cold
                    # path below answers the request.
                    pass
                else:
                    self._store(key, output, canonical)
                    return output

        output = vmecpp.run(
            vmecpp.VmecInput.model_validate(input), magnetic_field, **run_kwargs
        )
        self._store(key, output, canonical)
        return output

    def submit(
        self,
        input: VmecInput,
        magnetic_field: MagneticFieldResponseTable | None = None,
        *,
        verify: bool = True,
        max_threads: int | None = None,
        verbose: bool | int | None = None,
    ) -> concurrent.futures.Future[VmecOutput]:
        """:meth:`solve` on a background thread.

        Returns a :class:`concurrent.futures.Future` resolving to the
        :class:`VmecOutput`. The solver releases the GIL, so submitted solves
        run concurrently with the caller, and with each other up to
        ``max_parallel_solves``. The input is copied at submission time, so
        the caller may mutate its own instance afterwards.
        """
        with self._lock:
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._max_parallel_solves,
                    thread_name_prefix="vmecpp-cache",
                )
            executor = self._executor
        return executor.submit(
            self.solve,
            input.model_copy(deep=True),
            magnetic_field,
            verify=verify,
            max_threads=max_threads,
            verbose=verbose,
        )

    def close(self) -> None:
        """Wait for submitted solves and release the worker threads."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self) -> "EquilibriumCache":  # noqa: PYI034
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
