"""
Case 5: 3D 정사각 덕트 Poiseuille 유동 검증.

3D 정사각 단면 덕트, 층류 정상 유동.
해석해 (무한 급수)와 수치해를 비교한다.
격자: 20×8×8 셀
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator_3d import generate_3d_duct_mesh
from models.single_phase import SIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def analytical_3d_duct(y: np.ndarray, z: np.ndarray,
                        H: float, W: float,
                        dpdx: float, mu: float,
                        n_terms: int = 20) -> np.ndarray:
    """
    3D 정사각 덕트 Poiseuille 해석해 (무한 급수 근사).

    u(y,z) = (16*a^2 / (mu*pi^3)) * (-dpdx) *
              Σ_{n=0}^∞ (-1)^n / (2n+1)^3 *
              [1 - cosh((2n+1)*pi*z/a) / cosh((2n+1)*pi*b/(2a))] *
              cos((2n+1)*pi*y/a)

    여기서 a = H, b = W (단면이 -H/2..H/2, -W/2..W/2).
    입력은 y ∈ [0,H], z ∈ [0,W]이므로 좌표 변환 필요.
    """
    # 좌표를 대칭 중심 기준으로 변환
    y_c = y - H / 2.0  # [-H/2, H/2]
    z_c = z - W / 2.0  # [-W/2, W/2]
    a = H / 2.0
    b = W / 2.0

    u = np.zeros_like(y)

    # 간단한 2D Poiseuille 근사: 평행판 해석해에 보정 계수 적용
    # 정확한 급수 해
    G = -dpdx
    for n_idx in range(n_terms):
        lam = (2 * n_idx + 1) * np.pi / (2 * a)
        sign = (-1) ** n_idx
        coeff = sign / (2 * n_idx + 1) ** 3

        cos_term = np.cos((2 * n_idx + 1) * np.pi * y_c / (2 * a))
        cosh_num = np.cosh(lam * z_c)
        cosh_den = np.cosh(lam * b)

        # 오버플로 방지
        ratio = np.where(cosh_den > 1e200, 0.0, cosh_num / np.maximum(cosh_den, 1e-30))

        u += coeff * (1.0 - ratio) * cos_term

    u *= 4.0 * G * a**2 / (mu * np.pi**3)
    return np.maximum(u, 0.0)


def run_case5(results_dir: str = "results",
              figures_dir: str = "figures") -> dict:
    """
    3D 덕트 Poiseuille 검증 실행.

    Returns
    -------
    result : {'L2_error': float, 'converged': bool, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 5: 3D Duct Poiseuille 유동 검증")
    print("=" * 60)

    # 파라미터
    Lx = 2.0       # 덕트 길이
    Ly = 0.1       # 높이
    Lz = 0.1       # 폭
    nx, ny, nz = 20, 8, 8
    rho = 1.0
    mu = 0.01
    dpdx = -1.0

    # 해석해 최대 속도 (중심부)
    u_max_2d = -dpdx * Ly**2 / (8.0 * mu)  # 2D 근사 최대값 (참고용)
    print(f"  2D 근사 최대 속도: {u_max_2d:.6f} m/s")

    # 격자 생성
    print("  3D 격자 생성 중...")
    mesh = generate_3d_duct_mesh(Lx, Ly, Lz, nx, ny, nz)
    print(f"  {mesh.summary()}")

    # 솔버 설정
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 500
    solver.tol = 1e-4
    solver.alpha_u = 0.7
    solver.alpha_p = 0.3

    # 입구 포물선 프로파일 설정
    inlet_fids = mesh.boundary_patches.get('inlet', [])
    if inlet_fids:
        inlet_yz = np.array([mesh.faces[f].center for f in inlet_fids])
        u_inlet = analytical_3d_duct(inlet_yz[:, 1], inlet_yz[:, 2],
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
    y_num, z_num, u_num = [], [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[0] - x_target) < tol_x:
            y_num.append(cc[1])
            z_num.append(cc[2])
            u_num.append(solver.U.values[ci, 0])

    y_num = np.array(y_num)
    z_num = np.array(z_num)
    u_num = np.array(u_num)

    # 해석해 계산
    u_analytical = analytical_3d_duct(y_num, z_num, Ly, Lz, dpdx, mu)

    # L2 오차
    if len(u_analytical) > 0:
        L2_error = np.sqrt(np.mean((u_num - u_analytical)**2)) / max(np.max(np.abs(u_analytical)), 1e-15)
    else:
        L2_error = 1.0

    u_max_numerical = float(np.max(u_num)) if len(u_num) > 0 else 0.0
    u_max_anal = float(np.max(u_analytical)) if len(u_analytical) > 0 else 0.0
    print(f"  수치해 최대 속도: {u_max_numerical:.6f}")
    print(f"  해석해 최대 속도: {u_max_anal:.6f}")
    print(f"  L2 상대오차: {L2_error:.4e}")

    # 시각화: 중앙 z-평면에서의 속도 프로파일
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # z 중앙 근처 셀 선택
    z_mid = Lz / 2.0
    tol_z = Lz / nz * 1.5
    mask = np.abs(z_num - z_mid) < tol_z
    if np.any(mask):
        y_mid = y_num[mask]
        u_mid_num = u_num[mask]
        u_mid_anal = u_analytical[mask]

        idx = np.argsort(y_mid)
        ax = axes[0]
        y_fine = np.linspace(0, Ly, 100)
        z_fine = np.full_like(y_fine, z_mid)
        u_fine = analytical_3d_duct(y_fine, z_fine, Ly, Lz, dpdx, mu)
        ax.plot(u_fine, y_fine, 'r-', linewidth=2, label='해석해')
        ax.plot(u_mid_num[idx], y_mid[idx], 'bo', markersize=4, label='수치해 (FVM)')
        ax.set_xlabel('u [m/s]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'3D 덕트 속도 프로파일 (z={z_mid:.3f}m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 잔차 이력
    ax = axes[1]
    if result['residuals']:
        ax.semilogy(result['residuals'])
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('잔차')
    ax.set_title('수렴 이력')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case5_3d_duct.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    result_data = {
        'L2_error': L2_error,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'u_max_numerical': u_max_numerical,
        'u_max_analytical': u_max_anal,
        'figure_path': fig_path,
        'residuals': result['residuals']
    }

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case5')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'nx': nx, 'ny': ny, 'nz': nz,
              'rho': rho, 'mu': mu, 'dpdx': dpdx}
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
        print(f"  [VTU] case5 export skipped: {e}")

    return result_data


if __name__ == "__main__":
    result = run_case5()
    print(f"\n  결과 요약: L2 오차 = {result['L2_error']:.4e}")
    if result['L2_error'] < 0.05:
        print("  ✓ 검증 통과 (L2 < 5%)")
    else:
        print("  ✗ 검증 실패")
