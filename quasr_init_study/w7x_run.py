import json
import sys

sys.path.insert(0, "/tmp/quasr_init_study")
import vmecpp
from quasr_run import measure  # reuse (FTOL=1e-9, cap=1200)

vi = vmecpp.VmecInput.from_file("/home/jurasic/vmecpp/examples/data/w7x.json")
rows = []
print(f"=== w7x (mpol={vi.mpol}, ns=25) ===", flush=True)
for m in ["default", "zeno", "map2disc"]:
    r = measure(vi, 25, m)
    r["config"] = "w7x"
    rows.append(r)
    if r["status"] == "ok":
        print(
            f"  {m:9s} fsqt0={r['fsqt0']:.3e} niter={r['niter']:5d} cap={r['hit_cap']} conv={r['converged']} vol={r['volume']:.5f}",
            flush=True,
        )
    else:
        print(f"  {m:9s} {r['status']}", flush=True)
    open("/tmp/quasr_init_study/results_w7x.json", "w").write(
        json.dumps(rows, indent=2)
    )
