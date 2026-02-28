"""
Case 2: Lid-Driven Cavity 검증.

2D 정사각형, 상부 벽 이동 (U=1).
Re = 100, 400, 1000
Ghia et al. (1982) 벤치마크 데이터와 비교.
격자: 64×64 셀
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from mesh.mesh_generator import generate_cavity_mesh
from models.single_phase import SIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


# Ghia et al. (1982) 벤치마크 데이터
GHIA_DATA = {
    100: {
        'y': [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
              0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
              0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
        'u': [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
              -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
              0.68717, 0.73722, 0.78871, 0.84123, 1.0000]
    },
    400: {
        'y': [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
              0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
              0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
        'u': [0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
              -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
              0.55892, 0.61756, 0.68439, 0.75837, 1.0000]
    },
    1000: {
        'y': [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
              0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
              0.9531, 0.9609, 0.9688, 0.9766, 1.0000],
        'u': [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
              -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
              0.46604, 0.51117, 0.57492, 0.65928, 1.0000]
    }
}


def run_case2(re_list: list = None, n_grid: int = 32,
              results_dir: str = "results",
              figures_dir: str = "figures") -> dict:
    """
    Lid-Driven Cavity 검증 실행.

    Parameters
    ----------
    re_list : Reynolds 수 리스트 (기본: [100, 400])
    n_grid : 격자 크기 (n×n)

    Returns
    -------
    results : {Re: {'converged': bool, ...}}
    """
    if re_list is None:
        re_list = [100, 400]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 2: Lid-Driven Cavity 검증")
    print("=" * 60)

    U_lid = 1.0
    L = 1.0
    rho = 1.0

    all_results = {}

    for Re in re_list:
        print(f"\n  --- Re = {Re} ---")
        mu = rho * U_lid * L / Re

        # 격자 생성
        mesh = generate_cavity_mesh(L, n_grid)
        print(f"  격자: {n_grid}x{n_grid}, {mesh.n_cells} cells")

        # 솔버 설정
        solver = SIMPLESolver(mesh, rho=rho, mu=mu)
        solver.max_outer_iter = 3000
        solver.tol = 1e-4
        solver.alpha_u = 0.7
        solver.alpha_p = 0.3

        # 경계조건
        solver.set_velocity_bc('lid', 'dirichlet', [U_lid, 0.0])
        solver.set_velocity_bc('wall_bottom', 'dirichlet', [0.0, 0.0])
        solver.set_velocity_bc('wall_left', 'dirichlet', [0.0, 0.0])
        solver.set_velocity_bc('wall_right', 'dirichlet', [0.0, 0.0])
        solver.set_pressure_bc('lid', 'zero_gradient')
        solver.set_pressure_bc('wall_bottom', 'zero_gradient')
        solver.set_pressure_bc('wall_left', 'zero_gradient')
        solver.set_pressure_bc('wall_right', 'zero_gradient')

        # 해석
        print("  해석 중...")
        result = solver.solve_steady()
        print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

        # 중앙 수직선 (x=0.5) 속도 프로파일 추출
        x_mid = L / 2.0
        y_num, u_num, _ = solver.get_velocity_at_y(x_mid)

        if len(y_num) == 0:
            # fallback
            tol_x = L / n_grid * 1.5
            cells_near = [(ci, mesh.cells[ci].center)
                          for ci in range(mesh.n_cells)
                          if abs(mesh.cells[ci].center[0] - x_mid) < tol_x]
            if cells_near:
                y_num = np.array([c[1][1] for c in cells_near])
                u_num = np.array([solver.U.values[c[0], 0] for c in cells_near])
                idx = np.argsort(y_num)
                y_num = y_num[idx]
                u_num = u_num[idx]

        all_results[Re] = {
            'converged': result['converged'],
            'iterations': result['iterations'],
            'y_profile': y_num,
            'u_profile': u_num,
            'residuals': result['residuals']
        }

    # 시각화
    fig, axes = plt.subplots(1, len(re_list), figsize=(6 * len(re_list), 5))
    if len(re_list) == 1:
        axes = [axes]

    for idx, Re in enumerate(re_list):
        ax = axes[idx]
        res = all_results[Re]

        # 수치해
        ax.plot(res['u_profile'], res['y_profile'], 'b-', linewidth=1.5,
                label='수치해 (FVM)')

        # Ghia 데이터
        if Re in GHIA_DATA:
            ghia = GHIA_DATA[Re]
            ax.plot(ghia['u'], ghia['y'], 'ro', markersize=5,
                    label='Ghia et al. (1982)')

        ax.set_xlabel('u/U')
        ax.set_ylabel('y/L')
        ax.set_title(f'Lid-Driven Cavity, Re={Re}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case2_cavity.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {fig_path}")

    # 잔차 그래프
    fig, ax = plt.subplots(figsize=(8, 5))
    for Re in re_list:
        res = all_results[Re]
        if res['residuals']:
            ax.semilogy(res['residuals'], label=f'Re={Re}')
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('잔차')
    ax.set_title('Cavity 유동 수렴 이력')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_res_path = os.path.join(figures_dir, 'case2_cavity_residuals.png')
    plt.savefig(fig_res_path, dpi=150)
    plt.close()

    for Re in re_list:
        all_results[Re]['figure_path'] = fig_path

    # --- VTU / JSON export (last Re computed) ---
    last_Re = re_list[-1]
    mu_last = rho * U_lid * L / last_Re
    mesh_last = generate_cavity_mesh(L, n_grid)
    solver_last = SIMPLESolver(mesh_last, rho=rho, mu=mu_last)
    solver_last.max_outer_iter = 3000
    solver_last.tol = 1e-4
    solver_last.alpha_u = 0.7
    solver_last.alpha_p = 0.3
    solver_last.set_velocity_bc('lid', 'dirichlet', [U_lid, 0.0])
    solver_last.set_velocity_bc('wall_bottom', 'dirichlet', [0.0, 0.0])
    solver_last.set_velocity_bc('wall_left', 'dirichlet', [0.0, 0.0])
    solver_last.set_velocity_bc('wall_right', 'dirichlet', [0.0, 0.0])
    solver_last.set_pressure_bc('lid', 'zero_gradient')
    solver_last.set_pressure_bc('wall_bottom', 'zero_gradient')
    solver_last.set_pressure_bc('wall_left', 'zero_gradient')
    solver_last.set_pressure_bc('wall_right', 'zero_gradient')
    solver_last.solve_steady()
    case_dir = os.path.join(results_dir, 'case2')
    os.makedirs(case_dir, exist_ok=True)
    params = {'L': L, 'n_grid': n_grid, 're_list': re_list, 'rho': rho}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'velocity_x': solver_last.U.values[:, 0],
            'velocity_y': solver_last.U.values[:, 1],
            'pressure': solver_last.p.values,
        }
        export_mesh_to_vtu(mesh_last, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case2 export skipped: {e}")

    return all_results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results = run_case2(re_list=[100, 400], n_grid=32)
    for Re, res in results.items():
        print(f"\n  Re={Re}: 수렴={res['converged']}, 반복={res['iterations']}")
