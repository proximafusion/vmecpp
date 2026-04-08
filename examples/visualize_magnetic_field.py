"""Showcases how to (1) postprocess VMEC++ outputs to reconstruct the magnetic field
from Fourier coefficients, (2) transform field from toroidal to cylindrical and
Cartesian coordinates, and (3) visualize flux surface + magnetic field using pyVista.

Note, that this script requires pyvista as an additional dependency.
"""


def calculate_magnetic_field(
    vmec_output, j, theta, phi, output_coordinates="cartesian"
):
    """Evaluate magnetic field at a toroidal point described by `(j, theta, phi)`.

    Reference: J. Schilling (2026), "The Numerics of VMEC++", Proxima Fusion.

    Parameters
    ----------
    vmec_output : VmecOutput
        VMEC++ output from `vmecpp.run(...)`.
    j : int
        Index of the flux surface (between 0 and `vmec_output.wout.ns - 1`,
        where 0 is the magnetic axis and `vmec_output.wout.ns - 1` is the
        outermost flux surface).
    theta : float
        Polidal angle between `0` and `2*pi`.
    phi : float
        Toroidal angle between `0` and `2*pi`.
    output_coordinates : string
        Desired output coordinates (`cartesian` or `cylindrical`).

    Returns
    -------
    ndarray, ndarray, ndarray
        Position `X`, magnetic field vector `B`, and current density `J` in the
        desired coordinates.
    """

    # Fetch poloidal mode numbers: m for position, m_nyq for magnetic field
    xm = vmec_output.wout.xm
    xm_nyq = vmec_output.wout.xm_nyq

    # Fetch toroidal mode numbers: n * nfp for position, n_nyq * nfp for magnetic field
    xn = vmec_output.wout.xn
    xn_nyq = vmec_output.wout.xn_nyq

    # Fetch position Fourier coefficients
    rmnc = vmec_output.wout.rmnc
    zmns = vmec_output.wout.zmns

    # Fetch B^θ_mn and B^ζ_mn cos Fourier coefficients
    # NOTE: From Schilling 2026 VMEC++ white paper, Table 9.2, we have
    #       B^{θ,cos}_mn = bsupumnc and B^{ζ,cos}_mn = bsupvmnc
    bsupumnc = vmec_output.wout.bsupumnc
    bsupvmnc = vmec_output.wout.bsupvmnc

    # Arguments of Fourier series
    kernel = xm * theta - xn * phi
    kernel_nyq = xm_nyq * theta - xn_nyq * phi

    # Evaluate cos and sin
    cosk = np.cos(kernel)
    sink = np.sin(kernel)
    cosk_nyq = np.cos(kernel_nyq)

    # Position in cylindrical coordinates (Sec. 9.1)
    r = np.dot(rmnc[:, j], cosk)
    z = np.dot(zmns[:, j], sink)

    # Cylindrical position derivatives (Sec. 3.5)
    drdtheta = np.dot(rmnc[:, j], -xm * sink)
    drdphi = np.dot(rmnc[:, j], xn * sink)
    dzdtheta = np.dot(zmns[:, j], xm * cosk)
    dzdphi = np.dot(zmns[:, j], -xn * cosk)

    # Position in Cartesian coordinates (Sec. 3.7)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Construct contravariant components B^θ and B^ζ from Fourier series (Sec. 9.2)
    bsupu = np.dot(bsupumnc[:, j], cosk_nyq)
    bsupv = np.dot(bsupvmnc[:, j], cosk_nyq)

    # Construct cylindrical components from contraviariant components
    br = bsupu * drdtheta + bsupv * drdphi
    bphi = bsupv * r
    bz = bsupu * dzdtheta + bsupv * dzdphi

    # # Repeat the same process for the current density field
    # currumnc = vmec_output.wout.currumnc
    # currvmnc = vmec_output.wout.currvmnc

    # curru = np.dot(currumnc[:, j], cosk_nyq)
    # currv = np.dot(currvmnc[:, j], cosk_nyq)

    # currr = curru * drdtheta + currv * drdphi
    # currphi = currv * r
    # currz = curru * dzdtheta + currv * dzdphi

    if output_coordinates == "cylindrical":
        # Format outputs as vectors
        X = np.array([r, phi, z])  # Position in cylindrical coordinates
        B = np.array([br, bphi, bz])  # Magnetic field in cylindrical coordinates
        # J = np.array([currr, currphi, currz])     # Current density in cylindrical coordinates

    elif output_coordinates == "cartesian":
        # Magnetic field in Cartesian coordinates (Sec. 3.7)
        bx = br * np.cos(phi) - bphi * np.sin(phi)
        by = br * np.sin(phi) + bphi * np.cos(phi)

        # currx = currr * np.cos(phi) - currphi * np.sin(phi)
        # curry = currr * np.sin(phi) + currphi * np.cos(phi)

        # Format outputs as vectors
        X = np.array([x, y, z])  # Position in Cartesian coordinates
        B = np.array([bx, by, bz])  # Magnetic field in Cartesian coordinates
        # J = np.array([currx, curry, currz])       # Current density in Cartesian coordinates

    else:
        raise ValueError(
            f"Invalid `output_coordinates = {output_coordinates}`;"
            + " valid values are 'cartesian' and 'cylindrical'."
        )

    # return X, B, J
    return X, B


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    import pyvista as pv

    import vmecpp

    # ----------- RUN VMEC++ ---------------------------------------------------

    # Construct a VmecInput object, e.g. from a classic Fortran input file or VMEC++'s json format
    input_file = Path(__file__).parent / "data" / "input.w7x"
    vmec_input = vmecpp.VmecInput.from_file(input_file)

    # Run VMEC++:
    vmec_output = vmecpp.run(vmec_input)

    # ----------- CONSTRUCT MAGNETIC FIELD OVER A FLUX SURFACE -----------------

    # Number of flux surfaces, i.e., final radial resolution
    ns = vmec_output.wout.ns

    # Flux surface to process (ns - 1 is the outermost (last) flux surface, which is the plasma boundary)
    j = ns - 1

    # Resolution over the flux surface
    num_theta = 101
    num_phi = 181

    # Grid in theta and phi along the flux surface
    grid_theta = np.linspace(0.0, 2 * np.pi, num_theta, endpoint=True)
    grid_phi = np.linspace(0.0, 2 * np.pi, num_phi, endpoint=True)

    # Compute Cartesian coordinates of flux surface geometry and magnetic field
    x, y, z = (np.zeros([num_theta, num_phi]) for i in range(3))
    bx, by, bz = (np.zeros([num_theta, num_phi]) for i in range(3))
    jx, jy, jz = (np.zeros([num_theta, num_phi]) for i in range(3))

    for idx_theta, theta in enumerate(grid_theta):
        for idx_phi, phi in enumerate(grid_phi):
            # Compute
            # X, B, J = calculate_magnetic_field(vmec_output, j, theta, phi)
            X, B = calculate_magnetic_field(vmec_output, j, theta, phi)

            # Unpack coordinates
            x[idx_theta, idx_phi] = X[0]
            y[idx_theta, idx_phi] = X[1]
            z[idx_theta, idx_phi] = X[2]

            bx[idx_theta, idx_phi] = B[0]
            by[idx_theta, idx_phi] = B[1]
            bz[idx_theta, idx_phi] = B[2]

            # jx[idx_theta, idx_phi] = J[0]
            # jy[idx_theta, idx_phi] = J[1]
            # jz[idx_theta, idx_phi] = J[2]

    # ----------- VISUALIZE WITH pyVista ---------------------------------------

    # Create structured grid
    grid = pv.StructuredGrid(x, y, z)

    # Combine vector components
    B = np.stack((bx, by, bz), axis=2)
    J = np.stack((jx, jy, jz), axis=2)

    # Reshape to (n_points, 3)
    B = B.reshape(-1, 3, order="F")
    J = J.reshape(-1, 3, order="F")

    # Attach vector field
    grid["B"] = B
    grid["J"] = J

    # Initialize plotter
    plotter = pv.Plotter()

    # Add flux surface with B magnitude heat map
    plotter.add_mesh(
        grid,
        scalars="B",
        # cmap="plasma",            # Specify color scheme for heat map
        smooth_shading=True,
        specular=0.2,
        # show_edges=True,          # Show grid edges
        # line_width=0.5,
        # jupyter_backend='trame'   # Enable interactivity in Jupyter
    )

    # Add magnetic field glyphs
    glyphs = grid.glyph(orient="B", scale="B", factor=0.03)
    plotter.add_mesh(glyphs, color="blue")

    # Visualize
    plotter.show()

    # # Same thing with the current density
    # plotter.add_mesh(
    #     grid,
    #     scalars="J",
    #     smooth_shading=True,
    #     specular=0.2,
    # )

    # glyphs = grid.glyph(
    #     orient="J",
    #     scale="J",
    #     factor=0.03
    # )
    # plotter.add_mesh(glyphs, color="blue")

    # plotter.show()
