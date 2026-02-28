"""
ParaView 시각화 스크립트.

pvpython으로 각 검증 케이스의 VTU 파일을 렌더링하여 PNG 이미지 생성.
실행: "C:/Program Files/ParaView 6.0.1/bin/pvpython.exe" visualization/paraview_render.py
"""

import os
import sys

from paraview.simple import *

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def setup_2d_view(renderView, bounds):
    """2D 케이스용 카메라 설정."""
    xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    renderView.InteractionMode = '2D'
    renderView.CameraPosition = [cx, cy, 1.0]
    renderView.CameraFocalPoint = [cx, cy, 0.0]
    renderView.CameraViewUp = [0, 1, 0]
    renderView.CameraParallelProjection = 1
    ResetCamera()
    # 약간의 여백
    renderView.CameraParallelScale *= 1.05


def setup_3d_view(renderView, bounds):
    """3D 케이스용 아이소메트릭 카메라 설정."""
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    zmin, zmax = bounds[4], bounds[5]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    dist = max(dx, dy, dz) * 2.5
    renderView.CameraPosition = [cx + dist * 0.6, cy - dist * 0.4, cz + dist * 0.5]
    renderView.CameraFocalPoint = [cx, cy, cz]
    renderView.CameraViewUp = [0, 0, 1]
    ResetCamera()


def render_2d_case(vtu_path, field_name, output_path, title,
                   show_edges=False, preset='Jet'):
    """2D VTU 파일 렌더링."""
    if not os.path.exists(vtu_path):
        print(f"  SKIP: {vtu_path} not found")
        return False

    reader = XMLUnstructuredGridReader(FileName=[vtu_path])
    reader.UpdatePipeline()

    info = reader.GetDataInformation()
    bounds = info.GetBounds()

    # 필드 존재 확인
    cell_info = info.GetCellDataInformation()
    available = [cell_info.GetArrayInformation(i).GetName()
                 for i in range(cell_info.GetNumberOfArrays())]
    if field_name not in available:
        print(f"  SKIP: field '{field_name}' not in {available}")
        Delete(reader)
        return False

    renderView = CreateRenderView()
    renderView.ViewSize = [1200, 900]
    renderView.Background = [1.0, 1.0, 1.0]

    display = Show(reader, renderView)
    ColorBy(display, ('CELLS', field_name))
    if show_edges:
        display.SetRepresentationType('Surface With Edges')
        display.EdgeColor = [0.2, 0.2, 0.2]
    else:
        display.SetRepresentationType('Surface')

    lut = GetColorTransferFunction(field_name)
    lut.ApplyPreset(preset, True)
    colorBar = GetScalarBar(lut, renderView)
    colorBar.Title = title
    colorBar.Visibility = 1
    colorBar.TitleColor = [0, 0, 0]
    colorBar.LabelColor = [0, 0, 0]
    colorBar.TitleFontSize = 16
    colorBar.LabelFontSize = 14

    setup_2d_view(renderView, bounds)
    Render()
    SaveScreenshot(output_path, renderView, ImageResolution=[1200, 900])

    Delete(display)
    Delete(reader)
    Delete(renderView)
    print(f"  OK: {output_path}")
    return True


def render_3d_case(vtu_path, field_name, output_path, title,
                   slice_origin=None, slice_normal=None, preset='Jet'):
    """3D VTU 파일 렌더링 (외표면 + 선택적 단면)."""
    if not os.path.exists(vtu_path):
        print(f"  SKIP: {vtu_path} not found")
        return False

    reader = XMLUnstructuredGridReader(FileName=[vtu_path])
    reader.UpdatePipeline()

    info = reader.GetDataInformation()
    bounds = info.GetBounds()

    cell_info = info.GetCellDataInformation()
    available = [cell_info.GetArrayInformation(i).GetName()
                 for i in range(cell_info.GetNumberOfArrays())]
    if field_name not in available:
        print(f"  SKIP: field '{field_name}' not in {available}")
        Delete(reader)
        return False

    renderView = CreateRenderView()
    renderView.ViewSize = [1200, 900]
    renderView.Background = [1.0, 1.0, 1.0]

    if slice_origin and slice_normal:
        # 단면 표시
        sliceFilter = Slice(Input=reader)
        sliceFilter.SliceType = 'Plane'
        sliceFilter.SliceType.Origin = slice_origin
        sliceFilter.SliceType.Normal = slice_normal
        sliceFilter.UpdatePipeline()

        display = Show(sliceFilter, renderView)
        ColorBy(display, ('CELLS', field_name))
        display.SetRepresentationType('Surface')

        # 외곽선도 표시 (반투명)
        outline = Show(reader, renderView)
        outline.SetRepresentationType('Wireframe')
        outline.AmbientColor = [0.5, 0.5, 0.5]
        outline.DiffuseColor = [0.5, 0.5, 0.5]
        outline.Opacity = 0.15
    else:
        display = Show(reader, renderView)
        ColorBy(display, ('CELLS', field_name))
        display.SetRepresentationType('Surface')

    lut = GetColorTransferFunction(field_name)
    lut.ApplyPreset(preset, True)
    colorBar = GetScalarBar(lut, renderView)
    colorBar.Title = title
    colorBar.Visibility = 1
    colorBar.TitleColor = [0, 0, 0]
    colorBar.LabelColor = [0, 0, 0]
    colorBar.TitleFontSize = 16
    colorBar.LabelFontSize = 14

    setup_3d_view(renderView, bounds)
    Render()
    SaveScreenshot(output_path, renderView, ImageResolution=[1200, 900])

    # Cleanup
    for src in GetSources().values():
        Delete(src)
    Delete(renderView)
    print(f"  OK: {output_path}")
    return True


def render_all():
    """전체 15개 케이스 렌더링."""
    print("=" * 60)
    print("  ParaView VTU Rendering (15 Cases)")
    print("=" * 60)

    count = 0

    # Case 1: Poiseuille — velocity_x
    print("\nCase 1: Poiseuille")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case1", "mesh.vtu"),
        "velocity_x",
        os.path.join(FIGURES_DIR, "pv_case1_velocity.png"),
        "velocity_x [m/s]", show_edges=True):
        count += 1

    # Case 2: Cavity — velocity_x
    print("\nCase 2: Cavity")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case2", "mesh.vtu"),
        "velocity_x",
        os.path.join(FIGURES_DIR, "pv_case2_velocity.png"),
        "velocity_x [m/s]", show_edges=True):
        count += 1

    # Case 3: CHT — temperature
    print("\nCase 3: CHT")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case3", "mesh.vtu"),
        "temperature",
        os.path.join(FIGURES_DIR, "pv_case3_temperature.png"),
        "Temperature [K]", preset='Cool to Warm'):
        count += 1

    # Case 4: Bubble Column — alpha_g
    print("\nCase 4: Bubble Column")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case4", "mesh.vtu"),
        "alpha_g",
        os.path.join(FIGURES_DIR, "pv_case4_alpha.png"),
        "Gas Volume Fraction", preset='Cool to Warm'):
        count += 1

    # Case 5: 3D Duct — velocity_x (slice at z=Lz/2)
    print("\nCase 5: 3D Duct")
    if render_3d_case(
        os.path.join(RESULTS_DIR, "case5", "mesh.vtu"),
        "velocity_x",
        os.path.join(FIGURES_DIR, "pv_case5_velocity.png"),
        "velocity_x [m/s]",
        slice_origin=[0.5, 0.05, 0.025],
        slice_normal=[0, 0, 1]):
        count += 1

    # Case 6: MUSCL — phi_muscl
    print("\nCase 6: MUSCL")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case6", "mesh.vtu"),
        "phi_muscl",
        os.path.join(FIGURES_DIR, "pv_case6_muscl.png"),
        "phi (MUSCL)", preset='Cool to Warm'):
        count += 1

    # Case 7: Triangle Mesh — velocity_x + edges
    print("\nCase 7: Triangle Mesh")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case7", "mesh.vtu"),
        "velocity_x",
        os.path.join(FIGURES_DIR, "pv_case7_velocity.png"),
        "velocity_x [m/s]", show_edges=True):
        count += 1

    # Case 8: MPI — partition_id
    print("\nCase 8: MPI Partition")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case8", "mesh.vtu"),
        "partition_id",
        os.path.join(FIGURES_DIR, "pv_case8_partition.png"),
        "Partition ID", show_edges=True, preset='Set3'):
        count += 1

    # Case 9: Phase Change — temperature
    print("\nCase 9: Phase Change")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case9", "mesh.vtu"),
        "temperature",
        os.path.join(FIGURES_DIR, "pv_case9_temperature.png"),
        "Temperature [K]", preset='Cool to Warm'):
        count += 1

    # Case 10: Reaction — concentration_A
    print("\nCase 10: Reaction")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case10", "mesh.vtu"),
        "concentration_A",
        os.path.join(FIGURES_DIR, "pv_case10_concentration.png"),
        "Concentration A [mol/m3]", preset='Viridis (matplotlib)'):
        count += 1

    # Case 11: Radiation — G_radiation
    print("\nCase 11: Radiation")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case11", "mesh.vtu"),
        "G_radiation",
        os.path.join(FIGURES_DIR, "pv_case11_radiation.png"),
        "G radiation [W/m2]", preset='Cool to Warm'):
        count += 1

    # Case 12: AMR — temperature + edges
    print("\nCase 12: AMR")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case12", "mesh.vtu"),
        "temperature",
        os.path.join(FIGURES_DIR, "pv_case12_amr.png"),
        "Temperature [K]", show_edges=True, preset='Cool to Warm'):
        count += 1

    # Case 13: GPU — T (largest mesh)
    print("\nCase 13: GPU Benchmark")
    if render_2d_case(
        os.path.join(RESULTS_DIR, "case13_largest.vtu"),
        "T",
        os.path.join(FIGURES_DIR, "pv_case13_solution.png"),
        "Temperature [K]", preset='Cool to Warm'):
        count += 1

    # Case 14: 3D Cavity — u (slice at z=0.5)
    print("\nCase 14: 3D Cavity")
    if render_3d_case(
        os.path.join(RESULTS_DIR, "case14_3d_cavity.vtu"),
        "u",
        os.path.join(FIGURES_DIR, "pv_case14_velocity.png"),
        "u-velocity [m/s]",
        slice_origin=[0.5, 0.5, 0.5],
        slice_normal=[0, 0, 1]):
        count += 1

    # Case 15: 3D Convection — temperature (slice at y=0.5)
    print("\nCase 15: 3D Natural Convection")
    if render_3d_case(
        os.path.join(RESULTS_DIR, "case15", "mesh.vtu"),
        "temperature",
        os.path.join(FIGURES_DIR, "pv_case15_temperature.png"),
        "Temperature",
        slice_origin=[0.5, 0.5, 0.25],
        slice_normal=[0, 1, 0],
        preset='Cool to Warm'):
        count += 1

    print(f"\n{'=' * 60}")
    print(f"  완료: {count}/15 이미지 생성")
    print(f"{'=' * 60}")
    return count


if __name__ == "__main__":
    render_all()
