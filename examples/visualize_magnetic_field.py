"""Showcases how to (1) postprocess VMEC++ outputs to reconstruct the magnetic 
field from Fourier coefficients, (2) transform field from toroidal to 
cylindrical and Cartesian coordinates, and (3) visualize flux surface + magnetic 
field using pyVista.

Note, that this script requires pyvista as an additional dependency.
"""

def calculate_magnetic_field(vmec_output, j, theta, phi, 
                                output_coordinates='cartesian'):
    """
    Evaluate magnetic field at a toroidal point described by `(j, theta, phi)`.

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
    ndarray, ndarray
        Position `X` and magnetic field value `B` in the desired coordinates.
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
    bsupthtmnc = vmec_output.wout.bsupumnc
    bsupztamnc = vmec_output.wout.bsupvmnc
    
    # Arguments of Fourier series
    kernel = xm * theta - xn * phi
    kernel_nyq = xm_nyq * theta - xn_nyq * phi
    
    # Position in cylindrical coordinates (Sec. 9.1)
    r = np.dot(rmnc[:, j], np.cos(kernel))
    z = np.dot(zmns[:, j], np.sin(kernel))
    
    # Cylindrical position derivatives (Sec. 3.5)
    drdtheta = np.dot(rmnc[:, j], -xm * np.sin(kernel))
    drdphi = np.dot(rmnc[:, j], xn * np.sin(kernel))
    dzdtheta = np.dot(zmns[:, j], xm * np.cos(kernel))
    dzdphi = np.dot(zmns[:, j], -xn * np.cos(kernel))
    
    # Position in Cartesian coordinates (Sec. 3.7)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Construct contravariant components B^θ and B^ζ from Fourier series (Sec. 9.2)
    bsuptht = np.dot(bsupthtmnc[:, j], np.cos(kernel_nyq))
    bsupzta = np.dot(bsupztamnc[:, j], np.cos(kernel_nyq))
    
    # Construct cylindrical components from contraviariant components
    br = bsuptht * drdtheta + bsupzta * drdphi
    bphi = bsupzta * r
    bz = bsuptht * dzdtheta + bsupzta * dzdphi
    
    # Magnetic field in Cartesian coordinates (Sec. 3.7)
    bx = br * np.cos(phi) - bphi * np.sin(phi)
    by = br * np.sin(phi) + bphi * np.cos(phi)

    # Format outputs as vectors
    X = np.array([x, y, z])                       # Position in Cartesian coordinates
    B = np.array([bx, by, bz])                    # Magnetic field in Cartesian coordinates

    Xcyl = np.array([r, phi, z])                  # Position in cylindrical coordinates
    Bcyl = np.array([br, bphi, bz])               # Magnetic field in cylindrical coordinates

    # Return X and B
    if output_coordinates == 'cartesian':
        return X, B
    
    elif output_coordinates == 'cylindrical':
        return Xcyl, Bcyl
    
    else:
        raise ValueError(f"Invalid `output_coordinates = {output_coordinates}`;"
                          + " valid values are 'cartesian' and 'cylindrical'.")
