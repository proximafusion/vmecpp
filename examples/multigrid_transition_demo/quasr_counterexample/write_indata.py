"""Write a PARVMEC INDATA file from a cached QUASR meta.json.

Usage: write_indata.py <cachedir> <ns_csv> <delt>  (extcur fixed to 1.0, R-mode mgrid)
"""

import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
ns_csv = sys.argv[2]
delt = sys.argv[3]
meta = json.loads((d / "meta.json").read_text())
mpol, ntor, nfp = meta["mpol"], meta["ntor"], meta["nfp"]
cid = meta["config_id"]
rbc = np.array(meta["rbc"])
zbs = np.array(meta["zbs"])
ns_list = ns_csv.split(",")


def fmt(x):
    return f"{x:.10E}"


L = [
    "&INDATA",
    " LASYM = .FALSE.",
    f" NFP = {nfp}",
    f" MPOL = {mpol}",
    f" NTOR = {ntor}",
    " NTHETA = 0",
    f" NZETA = {meta['nzeta']}",
    " NS_ARRAY = " + " ".join(ns_list),
    " FTOL_ARRAY = " + " ".join(["1.0E-9"] * len(ns_list)),
    " NITER_ARRAY = " + " ".join(["3000"] * len(ns_list)),
    " NSTEP = 1",
    f" DELT = {delt}",
    f" PHIEDGE = {fmt(meta['phiedge'])}",
    " GAMMA = 0.0",
    " PMASS_TYPE = 'power_series'",
    " AM = 0.0",
    " PRES_SCALE = 0.0",
    " NCURR = 1",
    " PCURR_TYPE = 'power_series'",
    " AC = 0.0",
    " CURTOR = 0.0",
    " LFREEB = .TRUE.",
    f" MGRID_FILE = 'mgrid_quasr{cid:07d}.nc'",
    " EXTCUR = 1.0",
    " NVACSKIP = 6",
]
raxis = [0.0] * (ntor + 1)
raxis[0] = meta["raxis_guess"]
L.append(" RAXIS_CC = " + " ".join(fmt(x) for x in raxis))
L.append(" ZAXIS_CS = " + " ".join(fmt(0.0) for _ in range(ntor + 1)))
for m in range(mpol):
    for j in range(2 * ntor + 1):
        n = j - ntor
        if rbc[m, j] != 0.0:
            L.append(f" RBC({n},{m}) = {fmt(rbc[m, j])}")
        if zbs[m, j] != 0.0:
            L.append(f" ZBS({n},{m}) = {fmt(zbs[m, j])}")
L += ["/", ""]
tag = "-".join(ns_list)
out = d / f"input.quasr{cid}_{tag}"
out.write_text("\n".join(L))
print(out)
