# ------------------------------------------------------------------------------
# VTK example of plotting a torus and mapping a scalar field onto its surface.
# The scalar field is independent of Z, here using R + r*cos(v).
# The code renders a fully 3D object with correct depth ordering.
# ------------------------------------------------------------------------------

# call me as follows:
# parallel python examples/plot_torus_vtk.py {} ::: /home/jons/results/vmec_w7x/movie_cache/vmecpp_w7x_*.h5

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import vtk

import vmecpp

if len(sys.argv) < 2:
    print("usage: %s vmecpp_out.h5" % (sys.argv[0]))

vmecpp_out_filename = Path(sys.argv[1])
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

jxb_gradp = np.reshape(oq.jxbout.jxb_gradp, [ns, nzeta, ntheta3])

# extend to full poloidal range
jxb_gradp_full = np.zeros([ns, nfp * nzeta, ntheta1])
jxb_gradp_full[:, :nzeta, :ntheta3] = jxb_gradp
jxb_gradp_full[:, :nzeta, ntheta3:] = np.roll(
    jxb_gradp[:, :, 1:-1][:, ::-1, ::-1], shift=1, axis=1
)

# extend to full toroidal range
for kp in range(1, nfp):
    jxb_gradp_full[:, kp * nzeta : (kp + 1) * nzeta, :] = jxb_gradp_full[:, :nzeta, :]


def create_vtk_lut_from_matplotlib(cmap_name="jet", num_colors=256):
    """Create a vtkLookupTable by sampling a Matplotlib colormap.

    :param cmap_name: Name of a Matplotlib colormap (e.g., 'jet', 'viridis', etc.).
    :param num_colors: Number of discrete samples in the lookup table.
    :return: A vtkLookupTable filled with RGBA entries from the chosen colormap.
    """
    # Create an empty lookup table in VTK
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(num_colors)
    lut.Build()

    # Get the specified colormap from Matplotlib
    cmap = plt.get_cmap(cmap_name, num_colors)

    # Fill the VTK lookup table by sampling the Matplotlib colormap
    for i in range(num_colors):
        fraction = i / (num_colors - 1)
        r, g, b, a = cmap(fraction)
        lut.SetTableValue(i, r, g, b, a)

    return lut


# Create a jet-like lookup table from Matplotlib
lut = create_vtk_lut_from_matplotlib("jet", num_colors=256)
# lut = create_vtk_lut_from_matplotlib("inferno", num_colors=256)

# Create the main objects for VTK
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(True)
render_window.AddRenderer(renderer)
render_window.SetAlphaBitPlanes(True)

# # fixed colorbar scale
# scalar_min = -6.0e6
# scalar_max = 2.0e5

# js = 47

# # allow for incomplete toroidal range for some flux surfaces
# num_toroidal = nfp * nzeta - 18
# # num_toroidal = nfp * nzeta

delta_k = 6

# all_js = np.array(np.unique(np.round(1 + np.power(np.linspace(0, 1, 20), 2) * 48)), dtype=int)

for i, js in enumerate(
    [
        # 1, 10, 25, 35, ns-3
        # 1,  2,  3,  6, 11, 16, 23, 31, 40, 49
        1,
        2,
        3,
        4,
        6,
        8,
        10,
        12,
        14,
        17,
        20,
        23,
        27,
        31,
        35,
        39,
        44,
        49,
    ]
):
    num_toroidal = nfp * nzeta - i * delta_k

    # Arrays to hold coordinates and scalars
    points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()

    # Build the torus in a parametric grid:
    #   u in [0, 2π), v in [0, 2π)
    # We'll index as i in [0, n), j in [0, n) and use modulo for wrap-around
    for l in range(ntheta1):
        # Param u goes around the central hole
        theta = 2.0 * np.pi * l / ntheta1
        for k in range(min(num_toroidal + 1, nfp * nzeta)):
            # Param v goes around the tube's cross-section
            phi = 2.0 * np.pi * k / (nfp * nzeta)

            kernel = oq.wout.xm * theta - oq.wout.xn * phi

            r = np.dot(oq.wout.rmnc[js, :], np.cos(kernel))
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = np.dot(oq.wout.zmns[js, :], np.sin(kernel))

            # Insert the point
            points.InsertNextPoint(x, z, y)

            # Define the scalar field: R + r*cos(v) independent of z
            # scalar_value = np.log10(abs(jxb_gradp_full[js, k, l]))
            scalar_value = jxb_gradp_full[js, k, l]
            scalars.InsertNextValue(scalar_value)

    # Create a vtkPolyData to store the geometry
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)

    # We need to define connectivity (which points form each polygon).
    # We'll create a mesh of quadrilaterals, each made of two triangles.
    # Let idx(i,j) = i*n + j in a 1D index. We'll wrap around with modulo.
    cells = vtk.vtkCellArray()

    def idx(l, k):
        return l * min(num_toroidal + 1, nfp * nzeta) + k

    for l in range(ntheta1):
        l2 = (l + 1) % ntheta1
        for k in range(num_toroidal):
            k2 = (k + 1) % (nfp * nzeta)

            # We can form two triangles, or a single quad cell.
            # Here we'll make one quad: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, idx(l, k))
            quad.GetPointIds().SetId(1, idx(l2, k))
            quad.GetPointIds().SetId(2, idx(l2, k2))
            quad.GetPointIds().SetId(3, idx(l, k2))
            cells.InsertNextCell(quad)

    poly_data.SetPolys(cells)

    # Attach the scalars to the polydata
    poly_data.GetPointData().SetScalars(scalars)

    # Create a lookup table so we can color the range of scalars
    scalar_min = scalars.GetRange()[0]
    scalar_max = scalars.GetRange()[1]

    # symmetrize
    val_max = max(abs(scalar_min), abs(scalar_max))
    scalar_min = -val_max
    scalar_max = val_max

    # Create a mapper for the polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.SetScalarModeToUsePointData()

    # The range in the data we want to map. We'll just fake a scalar range here;
    # in a real case, you'd have scalar data from the geometry or from a separate array.
    # We'll forcibly set a "ScalarRange" to demonstrate usage.
    # If you actually have point or cell scalars, set them on the data
    # and let the mapper pick it up.
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalar_min, scalar_max)

    # Create an actor using this mapper
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Add the actor to the renderer
    renderer.AddActor(actor)


# Optionally add a scalar bar to show the color mapping
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetTitle("MHD Force Residual")
scalar_bar.SetLookupTable(lut)
scalar_bar.SetNumberOfLabels(5)
scalar_bar.SetVerticalTitleSeparation(20)

# Place the scalar bar as a 2D overlay (no widget/interactor needed)
# By default, it should appear on the right side of the image
renderer.AddActor2D(scalar_bar)

# ------------------------------------------------------------------------------
# 4) Customize Font, Size, and Text Color
# ------------------------------------------------------------------------------
# Access the text properties for title and labels separately
title_text_prop = scalar_bar.GetTitleTextProperty()
label_text_prop = scalar_bar.GetLabelTextProperty()

# Change font family (options include SetFontFamilyToArial, SetFontFamilyToTimes, etc.)
title_text_prop.SetFontFamilyToArial()
label_text_prop.SetFontFamilyToArial()

# Change font sizes (the actual rendered size also depends on the overall image size)
title_text_prop.SetFontSize(24)
label_text_prop.SetFontSize(16)

# Change text color to black (0,0,0)
title_text_prop.SetColor(0, 0, 0)
label_text_prop.SetColor(0, 0, 0)

# If you want to make the title bold or italic, you can do:
title_text_prop.SetBold(False)
title_text_prop.SetItalic(False)

label_text_prop.SetBold(False)
label_text_prop.SetItalic(False)

# Adjust the bar ratio, width, or position if you need to
scalar_bar.SetBarRatio(0.2)
scalar_bar.SetWidth(0.1)
scalar_bar.SetHeight(0.6)
scalar_bar.SetPosition(0.82, 0.2)

# Adjust background color: transparent white
renderer.SetBackground(1.0, 1.0, 1.0)  # white
renderer.SetBackgroundAlpha(0.0)

# Make sure we see our torus nicely
renderer.ResetCamera()

# -------------------------------------------------------------------
# Adjust the camera's elevation and azimuth
# -------------------------------------------------------------------
camera = renderer.GetActiveCamera()

# Elevation is the angle above/below the view plane
camera.Elevation(20.0)  # tilt up by 30 degrees

# Azimuth is rotation around the scene (viewing down from above)
# camera.Azimuth(45.0)  # rotate camera by 45 degrees around the focal point
# camera.Azimuth(90.0 + 20.0)
camera.Azimuth(90.0 + 40.0)

# Dolly the camera to zoom in (factor > 1 => zoom in; factor < 1 => zoom out)
# camera.Dolly(2.0)  # e.g., 1.3 means 30% closer to the focal point
camera.Dolly(1.5)

# alternative to Dolly: change camera opening angle
# camera.SetViewAngle(20)  # smaller angle => more zoom

# Update the camera's clipping range (otherwise geometry can get clipped)
renderer.ResetCameraClippingRange()

# ------------------------------------------------------------------------------
# Remove default lights and add a new light at the camera position
# ------------------------------------------------------------------------------
# By default, VTK automatically creates lights. We turn that off:
renderer.AutomaticLightCreationOff()
renderer.RemoveAllLights()

# Create a new light
light = vtk.vtkLight()
light.SetLightTypeToSceneLight()

# Position the light where the camera is
light.SetPosition(camera.GetPosition())

# Orient the light toward the camera's focal point
light.SetFocalPoint(camera.GetFocalPoint())

# Optionally adjust light properties: color, intensity, etc.
light.SetColor(1.0, 1.0, 1.0)  # white light
light.SetIntensity(1.0)  # brightness factor (1.0 = default)
renderer.AddLight(light)

# ------------------------------------------------------------------------------
# Render off-screen and save to an image file
# ------------------------------------------------------------------------------
render_window.SetSize(1920, 1080)
render_window.Render()

# Capture RGBA from the render window
window_to_image = vtk.vtkWindowToImageFilter()
window_to_image.SetInput(render_window)
# Request an RGBA buffer (with alpha)
window_to_image.SetInputBufferTypeToRGBA()
# Make sure we read the back buffer (without any prior composited front buffer)
window_to_image.ReadFrontBufferOff()
window_to_image.Update()

# Write the image to a file
writer = vtk.vtkPNGWriter()
writer.SetFileName(vmecpp_out_filename.with_suffix(".png"))
writer.SetInputConnection(window_to_image.GetOutputPort())
# This tells the PNG writer to preserve the alpha channel
if hasattr(writer, "SetWriteAlphaChannel"):
    print("using alpha channel for transparency")
    writer.SetWriteAlphaChannel(True)
writer.Write()
