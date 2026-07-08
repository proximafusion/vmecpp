import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_visualize_example():
    example_path = (
        Path(__file__).parents[1] / "examples" / "visualize_magnetic_field.py"
    )
    spec = importlib.util.spec_from_file_location(
        "visualize_magnetic_field", example_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_density_divides_vmec_current_harmonics_by_jacobian():
    module = load_visualize_example()
    wout = SimpleNamespace(
        xm=np.array([0.0]),
        xn=np.array([0.0]),
        xm_nyq=np.array([0.0]),
        xn_nyq=np.array([0.0]),
        rmnc=np.array([[2.0]]),
        zmns=np.array([[0.0]]),
        bsupumnc=np.array([[0.0]]),
        bsupvmnc=np.array([[0.0]]),
        gmnc=np.array([[4.0]]),
        currumnc=np.array([[8.0]]),
        currvmnc=np.array([[12.0]]),
    )
    vmec_output = SimpleNamespace(wout=wout)

    _, _, current = module.calculate_magnetic_field(
        vmec_output, j=0, theta=0.0, phi=0.0
    )

    np.testing.assert_allclose(current, np.array([0.0, 6.0, 0.0]))
