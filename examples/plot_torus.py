# This script demonstrates how to plot a scalar field mapped onto a toroidal surface in 3D
# using matplotlib in Python. The scalar field is defined to be independent of the Z-coordinate.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

import vmecpp

cache_folder = Path("/home/jons/results/vmec_w7x/movie_cache")
if not Path.exists(cache_folder):
    raise RuntimeError(
        "Output cache folder "
        + cache_folder
        + " does not exist. Run convergence_movie_make_runs.py to generate it."
    )

plots_folder = cache_folder / "plots"
Path.mkdir(plots_folder, parents=True, exist_ok=True)

saved_steps = np.loadtxt(cache_folder / "saved_steps.dat", dtype=int)

saved_step = saved_steps[-1]

vmecpp_out_filename = cache_folder / f"vmecpp_w7x_{saved_step:04d}.h5"
if not Path.exists(vmecpp_out_filename):
    raise RuntimeError(
        "VMEC++ output file "
        + str(vmecpp_out_filename)
        + " does not exist. Run convergence_movie_make_runs.py to generate it."
    )

oq = vmecpp._vmecpp.OutputQuantities.load(vmecpp_out_filename)

ns = oq.wout.ns
nfp = oq.wout.nfp

ntheta1 = 2 * (oq.indata.ntheta // 2)
ntheta3 = ntheta1 // 2 + 1
nzeta = oq.indata.nzeta

theta_grid = np.linspace(0.0, 2.0 * np.pi, ntheta1 + 1, endpoint=True)
phi_grid = np.linspace(0.0, 2.0 * np.pi, nfp * nzeta + 1, endpoint=True)

theta, phi = np.meshgrid(theta_grid, phi_grid)

kernel = np.outer(oq.wout.xm, theta) - np.outer(oq.wout.xn, phi)


# Create a new figure for 3D plotting
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection="3d")

cmap = cm.jet
# cmap = cm.inferno

dk = 18

for i, js in enumerate([1, 10, 35, 48]):
    k_start = i * dk

    # compute corresponding flux surface geometry
    r = np.zeros([ntheta3, nzeta])
    z = np.zeros([ntheta3, nzeta])

    r = (oq.wout.rmnc[js, :] @ np.cos(kernel)).reshape([nfp * nzeta + 1, ntheta1 + 1])
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = (oq.wout.zmns[js, :] @ np.sin(kernel)).reshape([nfp * nzeta + 1, ntheta1 + 1])

    # MHD force residual
    jxb_gradp = np.reshape(oq.jxbout.jxb_gradp, [ns, nzeta, ntheta3])[js, :, :]

    # extend to full poloidal range
    jxb_gradp_full = np.zeros([nfp * nzeta + 1, ntheta1 + 1])
    jxb_gradp_full[:nzeta, :ntheta3] = jxb_gradp
    jxb_gradp_full[:nzeta, ntheta3:] = jxb_gradp[:, 1:][::-1, ::-1]

    # extend to full toroidal range
    for kp in range(1, nfp):
        jxb_gradp_full[kp * nzeta : (kp + 1) * nzeta, :] = jxb_gradp_full[:nzeta, :]

    # ensure periodicity
    jxb_gradp_full[-1, :] = jxb_gradp_full[0, :]

    X = x[k_start:, :]
    Y = y[k_start:, :]
    Z = z[k_start:, :]
    scalar_field = jxb_gradp_full[k_start:, :]

    # Normalize the scalar field values to a [0, 1] range
    # FIXME: normalize over all occuring values in plot
    norm = Normalize(scalar_field.min(), scalar_field.max())

    # Map the normalized scalar field through a colormap
    mapped_colors = cmap(norm(scalar_field))

    # Plot the torus surface and use the mapped colors
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=mapped_colors,
        linewidth=0,
        antialiased=False,
        alpha=1.0,  # ensure fully opaque
        zsort="average",  # or 'min' or 'max'
    )

    # Create a ScalarMappable for the colorbar using the same normalization and colormap
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])  # Dummy array for the colorbar

# Attach the colorbar to the same Axes to avoid "Unable to determine Axes" errors
fig.colorbar(m, ax=ax, shrink=0.6, aspect=10, label="Scalar Field Value")

# Set the viewpoint by specifying elevation and azimuth angles
# elev is the angle (in degrees) above the xy-plane
# azim is the angle (in degrees) around the z-axis
ax.view_init(elev=30, azim=45)

# Force a 1:1:1 aspect ratio by setting the axis limits to a cube.
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5

ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

plt.tight_layout()

# Display the final plot
plt.show()
