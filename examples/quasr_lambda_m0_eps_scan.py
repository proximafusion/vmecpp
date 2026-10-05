"""Free-boundary QUASR scan of the tridiagonal m = 0 lambda preconditioner shift.

    python quasr_lambda_m0_eps_scan.py <eps,eps,...> <ftol> <out.json> [config_id,...|all] [lambda_preconditioner_scale]

eps < 0 is the diagonal VMEC 8.52 preconditioner. Uses the configurations, resolution and
mgrid construction of tests/test_free_boundary_quasr.py.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_free_boundary_quasr as q

eps_list = [float(e) for e in sys.argv[1].split(",")]
ftol = float(sys.argv[2])
out_path = Path(sys.argv[3])
ids = (
    [int(i) for i in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(q.QUASR_IDS)
)
profiles = [
    q.Profile(name="vacuum", target_beta=0.0),
    q.Profile(name="beta1", target_beta=0.01),
    q.Profile(name="beta2_current", target_beta=0.02, current_fraction=0.02),
]
rows = []
for config_id in ids:
    config = q._load_config(config_id)
    for profile in profiles:
        for eps in eps_list:
            vmec_input = q._make_input(config, profile, q.NS_ARRAY)
            vmec_input.ftol_array = np.full(len(q.NS_ARRAY), ftol)
            vmec_input.lambda_precondition_checkerboard_terms = eps
            vmec_input.lambda_preconditioner_scale = scale
            t = time.time()
            row = {
                "config": config_id,
                "profile": profile.name,
                "eps": eps,
                "ftol": ftol,
            }
            try:
                w = vmecpp.run(
                    vmec_input,
                    magnetic_field=config.response,
                    verbose=False,
                    max_threads=4,
                ).wout
                row.update(
                    status="converged",
                    iterations=len(w.fsqt),
                    restarts=len(w.restart_reasons)
                    if hasattr(w, "restart_reasons")
                    else None,
                    volume=float(w.volume),
                    betatotal=float(w.betatotal),
                    iota_axis=float(w.iotaf[0]),
                    iota_edge=float(w.iotaf[-1]),
                )
            except RuntimeError as exc:
                row.update(status="failed", error=str(exc).splitlines()[0][:200])
            row["wall"] = round(time.time() - t, 2)
            rows.append(row)
            print(json.dumps(row), flush=True)
            out_path.write_text(json.dumps(rows, indent=1))
