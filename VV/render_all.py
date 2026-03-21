"""Render all VV case VTU files with ParaView."""
from paraview.simple import *
import os

VV = "C:/Users/user/twofluid_fvm/VV"

def render(vtu, png, field, title, mode="2D"):
    if not os.path.exists(vtu):
        print(f"  SKIP: {vtu} not found"); return
    reader = XMLUnstructuredGridReader(FileName=[vtu])
    reader.UpdatePipeline()
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1400, 800]; view.Background = [1, 1, 1]
    display = Show(reader, view)
    ColorBy(display, ('CELLS', field))
    display.SetScalarBarVisibility(view, True)
    display.RescaleTransferFunctionToDataRange()
    lut = GetColorTransferFunction(field)
    lut.ApplyPreset('Jet', True)
    text = Text(Text=title)
    td = Show(text, view); td.FontSize = 16; td.Color = [0,0,0]
    td.WindowLocation = 'Upper Center'
    if mode == "2D": view.InteractionMode = '2D'
    else:
        cam = view.GetActiveCamera(); cam.Elevation(25); cam.Azimuth(30)
    view.ResetCamera(); Render()
    os.makedirs(os.path.dirname(png), exist_ok=True)
    SaveScreenshot(png, view, ImageResolution=[1400, 800])
    Delete(text); Delete(reader); ResetSession()
    print(f"  Saved: {png}")

print("Case 01: Poiseuille")
render(f"{VV}/case01_poiseuille/results/poiseuille.vtu", f"{VV}/case01_poiseuille/figures/velocity.png", "velocity_magnitude", "Case 1: Poiseuille Flow — Velocity |U| [m/s]")
render(f"{VV}/case01_poiseuille/results/poiseuille.vtu", f"{VV}/case01_poiseuille/figures/pressure.png", "pressure", "Case 1: Poiseuille Flow — Pressure [Pa]")

print("Case 02: Cavity")
render(f"{VV}/case02_cavity/results/cavity.vtu", f"{VV}/case02_cavity/figures/velocity.png", "velocity_magnitude", "Case 2: Cavity Re=100 — Velocity |U| [m/s]")
render(f"{VV}/case02_cavity/results/cavity.vtu", f"{VV}/case02_cavity/figures/pressure.png", "pressure", "Case 2: Cavity Re=100 — Pressure [Pa]")

print("Case 04: Bubble")
render(f"{VV}/case04_bubble_rising/results/bubble.vtu", f"{VV}/case04_bubble_rising/figures/alpha_gas.png", "alpha_gas", "Case 4: Bubble — Gas Volume Fraction")

print("Case 03: CHT")
render(f"{VV}/case03_cht/results/cht.vtu", f"{VV}/case03_cht/figures/velocity.png", "velocity_magnitude", "Case 3: CHT — Velocity |U| [m/s]")
render(f"{VV}/case03_cht/results/cht.vtu", f"{VV}/case03_cht/figures/pressure.png", "pressure", "Case 3: CHT — Pressure [Pa]")

print("Case 05: 3D Duct")
render(f"{VV}/case05_3d_duct/results/duct_3d.vtu", f"{VV}/case05_3d_duct/figures/velocity.png", "velocity_magnitude", "Case 5: 3D Duct — Velocity |U| [m/s]")

print("Case 07: Unstructured")
render(f"{VV}/case07_unstructured/results/poiseuille_unstructured.vtu", f"{VV}/case07_unstructured/figures/velocity.png", "velocity_magnitude", "Case 7: Unstructured Mesh — Velocity |U| [m/s]")

print("Case 08: MPI")
render(f"{VV}/case08_mpi/results/mpi_channel.vtu", f"{VV}/case08_mpi/figures/velocity.png", "velocity_magnitude", "Case 8: MPI Parallel — Velocity |U| [m/s]")

print("Case 10: Reaction")
render(f"{VV}/case10_reaction/results/reaction.vtu", f"{VV}/case10_reaction/figures/velocity.png", "velocity_magnitude", "Case 10: Reaction Transport — Velocity |U| [m/s]")

print("Case 12: AMR")
render(f"{VV}/case12_amr/results/amr_cavity.vtu", f"{VV}/case12_amr/figures/velocity.png", "velocity_magnitude", "Case 12: AMR Cavity — Velocity |U| [m/s]")

print("Case 13: GPU")
render(f"{VV}/case13_gpu/results/gpu_cavity.vtu", f"{VV}/case13_gpu/figures/velocity.png", "velocity_magnitude", "Case 13: GPU Cavity Re=400 — Velocity |U| [m/s]")
render(f"{VV}/case13_gpu/results/gpu_cavity.vtu", f"{VV}/case13_gpu/figures/pressure.png", "pressure", "Case 13: GPU Cavity Re=400 — Pressure [Pa]")

print("Case 15: 3D Convection")
render(f"{VV}/case15_3d_convection/results/convection_3d.vtu", f"{VV}/case15_3d_convection/figures/velocity.png", "velocity_magnitude", "Case 15: 3D Convection — Velocity |U| [m/s]")

print("Case 17: Adaptive dt")
render(f"{VV}/case17_adaptive_dt/results/adaptive_dt.vtu", f"{VV}/case17_adaptive_dt/figures/velocity.png", "velocity_magnitude", "Case 17: Adaptive dt — Velocity |U| [m/s]")

print("Case 18: OpenMP")
render(f"{VV}/case18_openmp/results/openmp_cavity.vtu", f"{VV}/case18_openmp/figures/velocity.png", "velocity_magnitude", "Case 18: OpenMP Cavity — Velocity |U| [m/s]")

print("Case BFS")
render(f"{VV}/case_bfs/results/bfs.vtu", f"{VV}/case_bfs/figures/velocity.png", "velocity_magnitude", "BFS Re=800 — Velocity |U| [m/s]")
render(f"{VV}/case_bfs/results/bfs.vtu", f"{VV}/case_bfs/figures/pressure.png", "pressure", "BFS Re=800 — Pressure [Pa]")

print("\nDone!")
