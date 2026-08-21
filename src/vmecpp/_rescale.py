import numpy as np

from . import VmecOutput, run


def rescale(
    output: VmecOutput,
    b_scale: float,
    r_scale: float,
    scale_pressure: bool = True,
) -> VmecOutput:
    """Rescale the equilibrium state.

    This scales the underlying geometry and inputs to represent an equilibrium
    with a scaled major radius R -> r_scale * R and magnetic field B -> b_scale * B.

    Args:
        output: a converged VmecOutput instance.
        b_scale: factor to scale the magnetic field by.
        r_scale: factor to scale the major radius by.
        scale_pressure: whether to scale pressure to maintain force balance (default: True).

    Returns:
        A new VmecOutput object with all derived parameters properly rescaled.
    """
    # Scale INDATA parameters
    scaled_input = output.input.model_copy(deep=True)
    scaled_input.phiedge *= b_scale * (r_scale**2)

    if scale_pressure:
        scaled_input.pres_scale *= b_scale**2

    scaled_input.curtor *= b_scale * r_scale

    scaled_input.rbc *= r_scale
    scaled_input.zbs *= r_scale
    if scaled_input.rbs is not None:
        scaled_input.rbs *= r_scale
    if scaled_input.zbc is not None:
        scaled_input.zbc *= r_scale

    scaled_input.raxis_c *= r_scale
    scaled_input.zaxis_s *= r_scale
    if scaled_input.raxis_s is not None:
        scaled_input.raxis_s *= r_scale
    if scaled_input.zaxis_c is not None:
        scaled_input.zaxis_c *= r_scale

    # Force 0 iterations for the restart
    scaled_input.niter_array = np.zeros_like(scaled_input.niter_array)

    # Scale WOUT geometry
    scaled_wout = output.wout.model_copy(deep=True)
    scaled_wout.rmnc *= r_scale
    scaled_wout.zmns *= r_scale
    if scaled_wout.rmns is not None:
        scaled_wout.rmns *= r_scale
    if scaled_wout.zmnc is not None:
        scaled_wout.zmnc *= r_scale

    # Create intermediate VmecOutput for restart_from
    # (The C++ side only uses wout and indata for HotRestartState, so it's fine if the rest is unscaled)
    intermediate_output = output.model_copy(deep=True)
    intermediate_output.input = scaled_input
    intermediate_output.wout = scaled_wout

    # Call run with 0 iterations
    return run(scaled_input, restart_from=intermediate_output)
