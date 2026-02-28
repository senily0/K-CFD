"""
VTU 파일에서 ParaView 스타일 렌더링 그림을 생성하는 모듈.

meshio + matplotlib를 사용하여 VTU 셀 데이터를 시각화한다.
ParaView가 설치되지 않은 환경에서도 보고서용 그림을 생성할 수 있다.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation
import os

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


def _read_vtu(vtu_path):
    """VTU 파일을 읽어 points, cells, cell_data를 반환."""
    if not HAS_MESHIO:
        return None, None, None
    m = meshio.read(vtu_path)
    points = m.points
    # 모든 셀 블록 결합
    all_cells = []
    cell_data_combined = {}
    for block in m.cells:
        for c in block.data:
            all_cells.append(c)
    # cell_data 결합
    for key, blocks in m.cell_data.items():
        combined = np.concatenate(blocks)
        cell_data_combined[key] = combined
    return points, all_cells, cell_data_combined


def render_2d_field(vtu_path, field_name, output_path, title=None,
                    cmap='jet', figsize=(8, 6)):
    """
    2D VTU 파일의 셀 데이터를 컬러맵으로 렌더링.

    Parameters
    ----------
    vtu_path : VTU 파일 경로
    field_name : 시각화할 셀 데이터 필드 이름
    output_path : 출력 PNG 경로
    title : 그림 제목
    cmap : 컬러맵 이름
    """
    if not HAS_MESHIO or not os.path.exists(vtu_path):
        print(f"  [VTU Render] 건너뜀: {vtu_path}")
        return False

    m = meshio.read(vtu_path)
    points = m.points[:, :2]  # x, y만 사용

    # 셀 + 데이터 결합
    all_cell_nodes = []
    all_values = []
    offset = 0
    for i, block in enumerate(m.cells):
        for ci, cell_nodes in enumerate(block.data):
            all_cell_nodes.append(cell_nodes)
        if field_name in m.cell_data:
            all_values.extend(m.cell_data[field_name][i])

    if len(all_values) == 0:
        print(f"  [VTU Render] 필드 '{field_name}' 없음: {vtu_path}")
        return False

    values = np.array(all_values)

    # PolyCollection으로 렌더링
    polygons = []
    for cell_nodes in all_cell_nodes:
        verts = points[cell_nodes]
        polygons.append(verts)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pc = PolyCollection(polygons, array=values, cmap=cmap, edgecolors='face',
                        linewidths=0.1)
    ax.add_collection(pc)
    ax.autoscale_view()
    # For highly anisotropic meshes (e.g. narrow channels), use 'auto' aspect
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    if x_range > 0 and y_range / x_range > 10:
        ax.set_aspect('auto')
    else:
        ax.set_aspect('equal')

    cb = plt.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(field_name)

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def render_3d_slices(vtu_path, field_name, output_path, title=None,
                     cmap='jet', figsize=(12, 5)):
    """
    3D VTU 파일의 셀 데이터를 3개 절단면(xy, xz, yz)으로 렌더링.

    Parameters
    ----------
    vtu_path : VTU 파일 경로
    field_name : 시각화할 셀 데이터 필드 이름
    output_path : 출력 PNG 경로
    title : 그림 제목
    cmap : 컬러맵 이름
    """
    if not HAS_MESHIO or not os.path.exists(vtu_path):
        print(f"  [VTU Render] 건너뜀: {vtu_path}")
        return False

    m = meshio.read(vtu_path)
    points = m.points

    # 셀 중심점 + 데이터 결합
    centers = []
    all_values = []
    for i, block in enumerate(m.cells):
        for ci, cell_nodes in enumerate(block.data):
            center = np.mean(points[cell_nodes], axis=0)
            centers.append(center)
        if field_name in m.cell_data:
            all_values.extend(m.cell_data[field_name][i])

    if len(all_values) == 0:
        print(f"  [VTU Render] 필드 '{field_name}' 없음: {vtu_path}")
        return False

    centers = np.array(centers)
    values = np.array(all_values)

    # 3D 범위
    x_min, y_min, z_min = centers.min(axis=0)
    x_max, y_max, z_max = centers.max(axis=0)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    dx = (x_max - x_min) / max(len(set(np.round(centers[:, 0], 6))), 1) * 1.5
    dy = (y_max - y_min) / max(len(set(np.round(centers[:, 1], 6))), 1) * 1.5
    dz = (z_max - z_min) / max(len(set(np.round(centers[:, 2], 6))), 1) * 1.5

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # XY 절단면 (z = z_mid)
    mask_xy = np.abs(centers[:, 2] - z_mid) < dz
    ax = axes[0]
    if np.sum(mask_xy) > 0:
        sc = ax.scatter(centers[mask_xy, 0], centers[mask_xy, 1],
                        c=values[mask_xy], cmap=cmap, s=5, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'XY plane (z~{z_mid:.3f})')
    ax.set_aspect('equal')

    # XZ 절단면 (y = y_mid)
    mask_xz = np.abs(centers[:, 1] - y_mid) < dy
    ax = axes[1]
    if np.sum(mask_xz) > 0:
        sc = ax.scatter(centers[mask_xz, 0], centers[mask_xz, 2],
                        c=values[mask_xz], cmap=cmap, s=5, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f'XZ plane (y~{y_mid:.3f})')
    ax.set_aspect('equal')

    # YZ 절단면 (x = x_mid)
    mask_yz = np.abs(centers[:, 0] - x_mid) < dx
    ax = axes[2]
    if np.sum(mask_yz) > 0:
        sc = ax.scatter(centers[mask_yz, 1], centers[mask_yz, 2],
                        c=values[mask_yz], cmap=cmap, s=5, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title(f'YZ plane (x~{x_mid:.3f})')
    ax.set_aspect('equal')

    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def render_all_missing(results_dir="results", figures_dir="figures"):
    """
    ParaView 렌더링 그림이 없는 케이스에 대해 자동 생성.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # 2D 케이스 매핑: (vtu_path, field_name, output_name, title)
    cases_2d = [
        # Cases 1-12 (originally paraview_render.py, now meshio fallback)
        ('case1/mesh.vtu', 'velocity_x', 'pv_case1_velocity.png',
         'Case 1: Poiseuille - 속도 분포'),
        ('case2/mesh.vtu', 'velocity_x', 'pv_case2_velocity.png',
         'Case 2: Lid-Driven Cavity - 속도 분포'),
        ('case3/mesh.vtu', 'temperature', 'pv_case3_temperature.png',
         'Case 3: CHT - 온도 분포'),
        ('case4/mesh.vtu', 'alpha_g', 'pv_case4_alpha.png',
         'Case 4: 기포탑 - 기체 체적분율'),
        ('case6/mesh.vtu', 'phi_muscl', 'pv_case6_muscl.png',
         'Case 6: MUSCL/TVD - 필드 분포'),
        ('case7/mesh.vtu', 'velocity_x', 'pv_case7_velocity.png',
         'Case 7: 비정렬 격자 - 속도 분포'),
        ('case8/mesh.vtu', 'partition_id', 'pv_case8_partition.png',
         'Case 8: MPI - 영역 분할'),
        ('case9/mesh.vtu', 'temperature', 'pv_case9_temperature.png',
         'Case 9: Stefan 문제 - 온도 분포'),
        ('case10/mesh.vtu', 'concentration_A', 'pv_case10_concentration.png',
         'Case 10: 반응 - 농도 분포'),
        ('case11/mesh.vtu', 'G_radiation', 'pv_case11_radiation.png',
         'Case 11: 복사 - 복사 에너지 분포'),
        ('case12/mesh.vtu', 'temperature', 'pv_case12_amr.png',
         'Case 12: AMR - 온도 분포'),
        ('case13_largest.vtu', 'T', 'pv_case13_solution.png',
         'Case 13: GPU - 해 분포'),
        # Cases 16-25
        ('case16/mesh.vtu', 'phi', 'pv_case16_solution.png',
         'Case 16: Preconditioner 비교 - 해 분포'),
        ('case17/mesh.vtu', 'phi', 'pv_case17_solution.png',
         'Case 17: Adaptive dt - 해 분포'),
        ('case19/mesh.vtu', 'temperature', 'pv_case19_temperature.png',
         'Case 19: 비등/응축 - 온도 분포'),
        ('case20/mesh.vtu', 'void_fraction', 'pv_case20_void.png',
         'Case 20: Edwards Blowdown - 보이드율'),
        ('case21/mesh.vtu', 'temperature_liquid', 'pv_case21_temperature.png',
         'Case 21: 6-Eq 가열 채널 - 액체 온도'),
        ('case24/boiling_final.vtu', 'alpha_gas', 'pv_case24_boiling.png',
         'Case 24: 풀 비등 - 증기 체적분율'),
        ('case25/condensation_final.vtu', 'alpha_liquid', 'pv_case25_condensation.png',
         'Case 25: 막 응축 - 액체 체적분율'),
    ]

    # 3D 케이스 매핑
    cases_3d = [
        ('case5/mesh.vtu', 'velocity_x', 'pv_case5_velocity.png',
         'Case 5: 3D 덕트 - 속도 분포'),
        ('case14_3d_cavity.vtu', 'u', 'pv_case14_velocity.png',
         'Case 14: 3D Cavity - 속도 분포'),
        ('case15/mesh.vtu', 'temperature', 'pv_case15_temperature.png',
         'Case 15: 3D 자연대류 - 온도 분포'),
        ('case22/mesh.vtu', 'u', 'pv_case22_velocity.png',
         'Case 22: 혼합 격자 - u-속도 분포'),
        ('case23/flow_final.vtu', 'velocity_magnitude', 'pv_case23_velocity.png',
         'Case 23: 3D 과도 채널 - 속도 크기'),
    ]

    rendered = 0

    for vtu_rel, field, out_name, title in cases_2d:
        vtu_path = os.path.join(results_dir, vtu_rel)
        out_path = os.path.join(figures_dir, out_name)
        if os.path.exists(out_path):
            continue
        if not os.path.exists(vtu_path):
            # 첫 번째 필드로 시도
            continue
        print(f"  렌더링: {out_name} ...")
        # 필드 존재 확인 후 대체
        if HAS_MESHIO:
            m = meshio.read(vtu_path)
            available = list(m.cell_data.keys())
            if field not in available and len(available) > 0:
                field = available[0]
        ok = render_2d_field(vtu_path, field, out_path, title=title)
        if ok:
            rendered += 1
            print(f"    완료: {out_path}")

    for vtu_rel, field, out_name, title in cases_3d:
        vtu_path = os.path.join(results_dir, vtu_rel)
        out_path = os.path.join(figures_dir, out_name)
        if os.path.exists(out_path):
            continue
        if not os.path.exists(vtu_path):
            continue
        print(f"  렌더링: {out_name} ...")
        if HAS_MESHIO:
            m = meshio.read(vtu_path)
            available = list(m.cell_data.keys())
            if field not in available and len(available) > 0:
                field = available[0]
        ok = render_3d_slices(vtu_path, field, out_path, title=title)
        if ok:
            rendered += 1
            print(f"    완료: {out_path}")

    print(f"  총 {rendered}개 ParaView 스타일 렌더링 생성 완료")
    return rendered


if __name__ == "__main__":
    render_all_missing()
