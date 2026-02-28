"""
Case 1: Poiseuille 유동 검증.

2D 평행판 사이 정상 층류 유동.
해석해: u(y) = ΔP/(2μL) · y·(H-y)
격자: 직사각형, 50×20 셀
검증: 속도 프로파일 비교, L2 오차 계산
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator import generate_channel_mesh
from models.single_phase import SIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def analytical_poiseuille(y: np.ndarray, H: float, dpdx: float, mu: float) -> np.ndarray:
    """
    Poiseuille 유동 해석해.

    u(y) = -dpdx/(2μ) · y·(H-y)
    """
    return -dpdx / (2.0 * mu) * y * (H - y)


def run_case1(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """
    Poiseuille 유동 검증 실행.

    Returns
    -------
    result : {'L2_error': float, 'converged': bool, 'u_max_numerical': float,
              'u_max_analytical': float}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 1: Poiseuille 유동 검증")
    print("=" * 60)

    # 파라미터
    L = 1.0       # 채널 길이
    H = 0.1       # 채널 높이
    nx, ny = 50, 20
    rho = 1.0     # 밀도
    mu = 0.01     # 점성
    dpdx = -1.0   # 압력 기울기

    # 해석해 최대 속도
    u_max_analytical = -dpdx * H**2 / (8.0 * mu)
    print(f"  해석해 최대 속도: {u_max_analytical:.6f} m/s")

    # 격자 생성
    print("  격자 생성 중...")
    mesh = generate_channel_mesh(L, H, nx, ny)
    print(f"  {mesh.summary()}")

    # 솔버 설정
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 500
    solver.tol = 1e-5
    solver.alpha_u = 0.7
    solver.alpha_p = 0.3

    # 경계조건
    # 입구: 포물선 프로파일 설정
    inlet_fids = mesh.boundary_patches.get('inlet', [])
    if inlet_fids:
        inlet_y = np.array([mesh.faces[f].center[1] for f in inlet_fids])
        u_inlet = analytical_poiseuille(inlet_y, H, dpdx, mu)
        u_vec = np.zeros((len(inlet_fids), 2))
        u_vec[:, 0] = u_inlet
        solver.U.boundary_values['inlet'] = u_vec

    solver.set_velocity_bc('inlet', 'dirichlet')
    solver.set_velocity_bc('outlet', 'zero_gradient')
    solver.set_velocity_bc('wall_bottom', 'dirichlet', [0.0, 0.0])
    solver.set_velocity_bc('wall_top', 'dirichlet', [0.0, 0.0])
    solver.set_pressure_bc('inlet', 'zero_gradient')
    solver.set_pressure_bc('outlet', 'dirichlet', 0.0)
    solver.set_pressure_bc('wall_bottom', 'zero_gradient')
    solver.set_pressure_bc('wall_top', 'zero_gradient')

    # 초기 조건: 균일 유동
    solver.U.values[:, 0] = u_max_analytical * 0.5
    solver.U.values[:, 1] = 0.0

    # 해석 실행
    print("  해석 중...")
    result = solver.solve_steady()
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # 출구 단면 속도 프로파일 추출
    x_target = L * 0.8
    y_num, u_num, _ = solver.get_velocity_at_y(x_target)

    if len(y_num) == 0:
        # fallback: 모든 셀의 y좌표 대비 속도
        y_num = np.array([mesh.cells[ci].center[1] for ci in range(mesh.n_cells)
                          if abs(mesh.cells[ci].center[0] - x_target) < L / nx * 1.5])
        u_num = np.array([solver.U.values[ci, 0] for ci in range(mesh.n_cells)
                          if abs(mesh.cells[ci].center[0] - x_target) < L / nx * 1.5])
        idx = np.argsort(y_num)
        y_num = y_num[idx]
        u_num = u_num[idx]

    # 해석해
    u_analytical = analytical_poiseuille(y_num, H, dpdx, mu)

    # L2 오차
    if len(u_analytical) > 0:
        L2_error = np.sqrt(np.mean((u_num - u_analytical)**2)) / max(np.max(np.abs(u_analytical)), 1e-15)
    else:
        L2_error = 1.0

    u_max_numerical = float(np.max(u_num)) if len(u_num) > 0 else 0.0
    print(f"  수치해 최대 속도: {u_max_numerical:.6f}")
    print(f"  L2 상대오차: {L2_error:.4e}")

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 속도 프로파일 비교
    ax = axes[0]
    y_fine = np.linspace(0, H, 100)
    u_fine = analytical_poiseuille(y_fine, H, dpdx, mu)
    ax.plot(u_fine, y_fine, 'r-', linewidth=2, label='해석해')
    ax.plot(u_num, y_num, 'bo', markersize=4, label='수치해 (FVM)')
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Poiseuille 유동 속도 프로파일 (x={x_target})')
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
    fig_path = os.path.join(figures_dir, 'case1_poiseuille.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # 결과 저장
    result_data = {
        'L2_error': L2_error,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'u_max_numerical': u_max_numerical,
        'u_max_analytical': float(u_max_analytical),
        'y_profile': y_num,
        'u_numerical': u_num,
        'u_analytical': u_analytical,
        'figure_path': fig_path
    }

    np.savez(os.path.join(results_dir, 'case1_poiseuille.npz'),
             y=y_num, u_num=u_num, u_anal=u_analytical)

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case1')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': L, 'Ly': H, 'nx': nx, 'ny': ny, 'rho': rho, 'mu': mu, 'dpdx': dpdx}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'velocity_x': solver.U.values[:, 0],
            'velocity_y': solver.U.values[:, 1],
            'pressure': solver.p.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case1 export skipped: {e}")

    return result_data


if __name__ == "__main__":
    result = run_case1()
    print(f"\n  결과 요약: L2 오차 = {result['L2_error']:.4e}")
    if result['L2_error'] < 0.01:
        print("  ✓ 검증 통과 (L2 < 1%)")
    else:
        print("  ✗ 검증 실패")
