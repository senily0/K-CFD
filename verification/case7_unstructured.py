"""
Case 7: 비정렬 격자 (삼각형) Poiseuille 유동 검증.

삼각형 격자에서 2D Poiseuille 유동을 풀고 해석해와 비교.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from mesh.mesh_generator import generate_triangle_channel_mesh
from models.single_phase import SIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def analytical_poiseuille(y: np.ndarray, H: float,
                           dpdx: float, mu: float) -> np.ndarray:
    """2D Poiseuille 해석해: u(y) = -dpdx/(2*mu) * y * (H - y)."""
    return -dpdx / (2.0 * mu) * y * (H - y)


def run_case7(results_dir: str = "results",
              figures_dir: str = "figures") -> dict:
    """
    삼각형 격자 Poiseuille 검증.

    Returns
    -------
    result : {'L2_error': float, 'converged': bool, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 7: Unstructured (Triangle) Poiseuille Verification")
    print("=" * 60)

    # 파라미터 (등방성 격자: dx ≈ dy)
    Lx = 0.2
    Ly = 0.1
    nx = 20
    ny = 10
    rho = 1.0
    mu = 0.01
    dpdx = -1.0

    u_max = -dpdx * Ly**2 / (8.0 * mu)
    print(f"  해석해 최대 속도: {u_max:.6f} m/s")

    # 삼각형 격자 생성
    print("  삼각형 격자 생성 중...")
    mesh = generate_triangle_channel_mesh(Lx, Ly, nx, ny)
    print(f"  {mesh.summary()}")

    # 솔버 설정 (비정렬 격자용 보수적 이완)
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 3000
    solver.tol = 1e-4
    solver.alpha_u = 0.2
    solver.alpha_p = 0.05

    # 입구 포물선 프로파일
    ndim = getattr(mesh, 'ndim', 2)
    inlet_fids = mesh.boundary_patches.get('inlet', [])
    if inlet_fids:
        inlet_y = np.array([mesh.faces[f].center[1] for f in inlet_fids])
        u_inlet = analytical_poiseuille(inlet_y, Ly, dpdx, mu)
        u_vec = np.zeros((len(inlet_fids), ndim))
        u_vec[:, 0] = u_inlet
        solver.U.boundary_values['inlet'] = u_vec

    # 경계조건
    solver.set_velocity_bc('inlet', 'dirichlet')
    solver.set_velocity_bc('outlet', 'zero_gradient')
    for wall in ['wall_bottom', 'wall_top']:
        if wall in mesh.boundary_patches:
            solver.set_velocity_bc(wall, 'dirichlet',
                                    np.zeros(ndim).tolist())
    solver.set_pressure_bc('inlet', 'zero_gradient')
    solver.set_pressure_bc('outlet', 'dirichlet', 0.0)
    for wall in ['wall_bottom', 'wall_top']:
        if wall in mesh.boundary_patches:
            solver.set_pressure_bc(wall, 'zero_gradient')

    # 초기 조건
    solver.U.values[:, 0] = u_max * 0.5
    solver.U.values[:, 1] = 0.0

    # 해석
    print("  해석 중...")
    result = solver.solve_steady()
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # 출구 근처 속도 프로파일
    x_target = Lx * 0.8
    tol_x = Lx / nx * 2.0
    y_num, u_num = [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[0] - x_target) < tol_x:
            y_num.append(cc[1])
            u_num.append(solver.U.values[ci, 0])

    y_num = np.array(y_num)
    u_num = np.array(u_num)

    # 해석해
    u_analytical = analytical_poiseuille(y_num, Ly, dpdx, mu)

    # L2 오차
    if len(u_analytical) > 0:
        L2_error = np.sqrt(np.mean((u_num - u_analytical)**2)) / max(
            np.max(np.abs(u_analytical)), 1e-15)
    else:
        L2_error = 1.0

    print(f"  L2 상대오차: {L2_error:.4e}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 속도 프로파일
    ax = axes[0]
    y_fine = np.linspace(0, Ly, 100)
    u_fine = analytical_poiseuille(y_fine, Ly, dpdx, mu)
    ax.plot(u_fine, y_fine, 'r-', linewidth=2, label='Analytical')
    idx = np.argsort(y_num)
    ax.plot(u_num[idx], y_num[idx], 'bo', markersize=4,
            label='Numerical (Triangle FVM)')
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('y [m]')
    ax.set_title('Triangle Mesh Poiseuille Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 수렴 이력
    ax = axes[1]
    if result['residuals']:
        ax.semilogy(result['residuals'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case7_unstructured.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case7')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'nx': nx, 'ny': ny, 'rho': rho, 'mu': mu, 'dpdx': dpdx}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'velocity_x': solver.U.values[:, 0],
            'velocity_y': solver.U.values[:, 1],
            'pressure': solver.p.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case7 export skipped: {e}")

    return {
        'L2_error': L2_error,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'n_cells': mesh.n_cells,
        'figure_path': fig_path,
        'residuals': result['residuals'],
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case7()
    print(f"\n  결과: L2 오차 = {result['L2_error']:.4e} ({result['n_cells']} cells)")
    if result['L2_error'] < 0.10:
        print("  V 검증 통과 (L2 < 10%)")
    else:
        print("  X 검증 실패")
