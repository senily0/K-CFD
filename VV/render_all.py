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

print("Case BFS")
render(f"{VV}/case_bfs/results/bfs.vtu", f"{VV}/case_bfs/figures/velocity.png", "velocity_magnitude", "BFS Re=800 — Velocity |U| [m/s]")
render(f"{VV}/case_bfs/results/bfs.vtu", f"{VV}/case_bfs/figures/pressure.png", "pressure", "BFS Re=800 — Pressure [Pa]")

print("\nDone!")
