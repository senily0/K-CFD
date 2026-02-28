"""
Case 22: 혼합 격자(Hex/Tet) Poiseuille 유동 검증.

3D 덕트 Poiseuille 유동을 hex+tet 혼합 격자에서 해석하고,
Case 5(순수 hex)와 동일한 해석해 대비 L2 오차를 비교한다.

검증 항목:
  - L2 오차 < 0.10 (tet 영역 허용 오차 감안)
  - hex/tet 인터페이스에서 해의 연속성
  - 수렴성
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.hybrid_mesh_generator import generate_hybrid_hex_tet_mesh
from models.single_phase import SIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def analytical_3d_duct(y, z, H, W, dpdx, mu, n_terms=20):
    """3D 정사각 덕트 Poiseuille 해석해 (Case 5와 동일)."""
    y_c = y - H / 2.0
    z_c = z - W / 2.0
    a = H / 2.0
    b = W / 2.0

    u = np.zeros_like(y)
    G = -dpdx

    for n_idx in range(n_terms):
        lam = (2 * n_idx + 1) * np.pi / (2 * a)
        sign = (-1) ** n_idx
        coeff = sign / (2 * n_idx + 1) ** 3

        cos_term = np.cos((2 * n_idx + 1) * np.pi * y_c / (2 * a))
        cosh_num = np.cosh(lam * z_c)
        cosh_den = np.cosh(lam * b)

        ratio = np.where(cosh_den > 1e200, 0.0,
                         cosh_num / np.maximum(cosh_den, 1e-30))
        u += coeff * (1.0 - ratio) * cos_term

    u *= 4.0 * G * a**2 / (mu * np.pi**3)
    return np.maximum(u, 0.0)


def run_case22(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    혼합 격자 Poiseuille 검증 실행.

    Returns
    -------
    result : dict with 'L2_error', 'converged', etc.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 22: 혼합 격자(Hex/Tet) Poiseuille 유동 검증")
    print("=" * 60)

    # 파라미터 (Case 5와 동일 물리)
    Lx = 2.0
    Ly = 0.1
    Lz = 0.1
    nx, ny, nz = 20, 6, 6
    rho = 1.0
    mu = 0.01
    dpdx = -1.0
    tet_fraction = 0.5  # 후반 50%가 tet

    u_max_2d = -dpdx * Ly**2 / (8.0 * mu)
    print(f"  2D 근사 최대 속도: {u_max_2d:.6f} m/s")
    print(f"  혼합 격자: hex {int(nx*(1-tet_fraction))} + tet {int(nx*tet_fraction)} layers (x방향)")

    # 혼합 격자 생성
    print("  혼합 격자 생성 중...")
    mesh = generate_hybrid_hex_tet_mesh(
        Lx, Ly, Lz, nx, ny, nz,
        tet_fraction=tet_fraction)
    print(f"  {mesh.summary()}")

    # hex/tet 셀 수
    n_hex = len(mesh.cell_zones.get('hex', []))
    n_tet = len(mesh.cell_zones.get('tet', []))
    print(f"  Hex 셀: {n_hex}, Tet 셀: {n_tet}, 총: {mesh.n_cells}")

    # 솔버 설정
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 500
    solver.tol = 1e-4
    solver.alpha_u = 0.7
    solver.alpha_p = 0.3

    # 입구 포물선 프로파일 설정
    inlet_fids = mesh.boundary_patches.get('inlet', [])
    if inlet_fids:
        inlet_coords = np.array([mesh.faces[f].center for f in inlet_fids])
        u_inlet = analytical_3d_duct(inlet_coords[:, 1], inlet_coords[:, 2],
                                      Ly, Lz, dpdx, mu)
        u_vec = np.zeros((len(inlet_fids), 3))
        u_vec[:, 0] = u_inlet
        solver.U.boundary_values['inlet'] = u_vec

    # 경계조건
    solver.set_velocity_bc('inlet', 'dirichlet')
    solver.set_velocity_bc('outlet', 'zero_gradient')
    for wall in ['wall_bottom', 'wall_top', 'wall_front', 'wall_back']:
        solver.set_velocity_bc(wall, 'dirichlet', [0.0, 0.0, 0.0])
    solver.set_pressure_bc('inlet', 'zero_gradient')
    solver.set_pressure_bc('outlet', 'dirichlet', 0.0)
    for wall in ['wall_bottom', 'wall_top', 'wall_front', 'wall_back']:
        solver.set_pressure_bc(wall, 'zero_gradient')

    # 초기 조건
    solver.U.values[:, 0] = u_max_2d * 0.3
    solver.U.values[:, 1] = 0.0
    solver.U.values[:, 2] = 0.0

    # 해석 실행
    print("  해석 중...")
    result = solver.solve_steady()
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # 출구 근처 단면 속도 프로파일 추출
    x_target = Lx * 0.8
    tol_x = Lx / nx * 1.5
    y_num, z_num, u_num, cell_zone_flag = [], [], [], []

    nx_hex = max(1, int(nx * (1.0 - tet_fraction)))
    x_interface = nx_hex * (Lx / nx)

    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[0] - x_target) < tol_x:
            y_num.append(cc[1])
            z_num.append(cc[2])
            u_num.append(solver.U.values[ci, 0])
            cell_zone_flag.append('tet' if cc[0] > x_interface else 'hex')

    y_num = np.array(y_num)
    z_num = np.array(z_num)
    u_num = np.array(u_num)

    # 해석해 계산
    u_analytical = analytical_3d_duct(y_num, z_num, Ly, Lz, dpdx, mu)

    # L2 오차
    if len(u_analytical) > 0 and np.max(np.abs(u_analytical)) > 1e-15:
        L2_error = np.sqrt(np.mean((u_num - u_analytical)**2)) / np.max(np.abs(u_analytical))
    else:
        L2_error = 1.0

    # hex/tet 영역별 L2 오차
    hex_mask = np.array([z == 'hex' for z in cell_zone_flag])
    tet_mask = np.array([z == 'tet' for z in cell_zone_flag])

    L2_hex = 0.0
    L2_tet = 0.0
    if np.any(hex_mask) and np.max(np.abs(u_analytical[hex_mask])) > 1e-15:
        L2_hex = np.sqrt(np.mean((u_num[hex_mask] - u_analytical[hex_mask])**2)) / \
                 np.max(np.abs(u_analytical[hex_mask]))
    if np.any(tet_mask) and np.max(np.abs(u_analytical[tet_mask])) > 1e-15:
        L2_tet = np.sqrt(np.mean((u_num[tet_mask] - u_analytical[tet_mask])**2)) / \
                 np.max(np.abs(u_analytical[tet_mask]))

    # 인터페이스 연속성 검사
    x_if = x_interface
    tol_if = Lx / nx * 2.0
    u_hex_if, u_tet_if = [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[0] - x_if) < tol_if:
            if cc[0] <= x_if:
                u_hex_if.append(solver.U.values[ci, 0])
            else:
                u_tet_if.append(solver.U.values[ci, 0])

    if u_hex_if and u_tet_if:
        interface_jump = abs(np.mean(u_hex_if) - np.mean(u_tet_if))
        interface_continuous = interface_jump < 0.1 * u_max_2d
    else:
        interface_jump = 0.0
        interface_continuous = True

    u_max_numerical = float(np.max(u_num)) if len(u_num) > 0 else 0.0
    u_max_anal = float(np.max(u_analytical)) if len(u_analytical) > 0 else 0.0

    print(f"  전체 L2 오차: {L2_error:.4e}")
    print(f"  Hex 영역 L2: {L2_hex:.4e}")
    print(f"  Tet 영역 L2: {L2_tet:.4e}")
    print(f"  인터페이스 점프: {interface_jump:.6f} (연속={interface_continuous})")

    # 시각화 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 속도 프로파일 (z 중앙)
    ax = axes[0, 0]
    z_mid = Lz / 2.0
    tol_z = Lz / nz * 1.5
    mask_z = np.abs(z_num - z_mid) < tol_z
    if np.any(mask_z):
        y_mid = y_num[mask_z]
        u_mid_num = u_num[mask_z]
        idx_sort = np.argsort(y_mid)

        y_fine = np.linspace(0, Ly, 100)
        z_fine = np.full_like(y_fine, z_mid)
        u_fine = analytical_3d_duct(y_fine, z_fine, Ly, Lz, dpdx, mu)

        ax.plot(u_fine, y_fine, 'r-', linewidth=2, label='해석해')
        ax.plot(u_mid_num[idx_sort], y_mid[idx_sort], 'bo', markersize=4,
                label='수치해 (Hybrid)')
        ax.set_xlabel('u [m/s]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'속도 프로파일 (z={z_mid:.3f}m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # (2) L2 오차 비교
    ax = axes[0, 1]
    labels = ['전체', 'Hex 영역', 'Tet 영역']
    errors = [L2_error, L2_hex, L2_tet]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(labels, errors, color=colors)
    ax.axhline(y=0.10, color='k', linestyle='--', alpha=0.5, label='허용 한계 (10%)')
    ax.set_ylabel('L2 상대 오차')
    ax.set_title('영역별 L2 오차')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # (3) 격자 시각화 (x-y 평면 z 중앙)
    ax = axes[1, 0]
    # 셀 중심 표시 (hex: 파랑, tet: 빨강)
    hex_cells = mesh.cell_zones.get('hex', [])
    tet_cells = mesh.cell_zones.get('tet', [])
    tol_z_vis = Lz / nz * 1.5
    for ci in hex_cells:
        cc = mesh.cells[ci].center
        if abs(cc[2] - z_mid) < tol_z_vis:
            ax.plot(cc[0], cc[1], 'bs', markersize=2, alpha=0.5)
    for ci in tet_cells:
        cc = mesh.cells[ci].center
        if abs(cc[2] - z_mid) < tol_z_vis:
            ax.plot(cc[0], cc[1], 'r^', markersize=1.5, alpha=0.5)
    ax.axvline(x=x_interface, color='k', linestyle='--', alpha=0.7, label='Hex/Tet 경계')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('격자 분포 (z 중앙면)')
    ax.legend()
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.3)

    # (4) 수렴 이력
    ax = axes[1, 1]
    if result['residuals']:
        ax.semilogy(result['residuals'])
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('잔차')
    ax.set_title('수렴 이력')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Case 22: 혼합 격자(Hex/Tet) Poiseuille', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case22_hybrid_mesh.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    result_data = {
        'L2_error': L2_error,
        'L2_hex': L2_hex,
        'L2_tet': L2_tet,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'u_max_numerical': u_max_numerical,
        'u_max_analytical': u_max_anal,
        'interface_continuous': interface_continuous,
        'interface_jump': float(interface_jump),
        'n_hex_cells': n_hex,
        'n_tet_cells': n_tet,
        'figure_path': fig_path,
        'residuals': result['residuals'],
    }

    # VTU / JSON export
    case_dir = os.path.join(results_dir, 'case22')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'nx': nx, 'ny': ny, 'nz': nz,
              'rho': rho, 'mu': mu, 'dpdx': dpdx, 'tet_fraction': tet_fraction}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'velocity_x': solver.U.values[:, 0],
            'velocity_y': solver.U.values[:, 1],
            'velocity_z': solver.U.values[:, 2],
            'pressure': solver.p.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case22 export skipped: {e}")

    return result_data


if __name__ == "__main__":
    result = run_case22()
    print(f"\n  L2 오차: {result['L2_error']:.4e}")
    if result['L2_error'] < 0.10:
        print("  ✓ 검증 통과 (L2 < 10%)")
    else:
        print("  ✗ 검증 실패")
