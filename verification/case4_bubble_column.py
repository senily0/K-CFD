"""
Case 4: 기포탑 이상유동 검증.

2D 직사각형 수조에 하부 중앙에서 공기 주입.
물-공기 시스템.
Two-Fluid 모델 + Schiller-Naumann 항력.
검증: 체적분율 분포, 기포 상승 패턴 정성적 비교.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator import generate_bubble_column_mesh
from models.two_fluid import TwoFluidSolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case4(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """
    기포탑 이상유동 검증 실행.

    Returns
    -------
    result : dict with simulation data
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 4: 기포탑 이상유동 검증")
    print("=" * 60)

    # 파라미터
    width = 0.15     # 수조 폭 [m]
    height = 0.45    # 수조 높이 [m]
    nx, ny = 8, 20  # 격자 수 (성능을 위해 축소)

    # 물-공기 물성치
    rho_l = 998.2
    rho_g = 1.225
    mu_l = 1.003e-3
    mu_g = 1.789e-5
    d_b = 0.005      # 기포 직경 5mm

    # 입구 조건
    alpha_g_inlet = 0.04  # 입구 기체 체적분율
    U_g_inlet = 0.1       # 입구 기체 속도 (상향) [m/s]

    # 격자 생성
    print("  격자 생성 중...")
    mesh = generate_bubble_column_mesh(width, height, nx, ny)
    print(f"  {mesh.summary()}")

    # 솔버 설정
    solver = TwoFluidSolver(mesh)
    solver.rho_l = rho_l
    solver.rho_g = rho_g
    solver.mu_l = mu_l
    solver.mu_g = mu_g
    solver.d_b = d_b
    solver.g = np.array([0.0, -9.81])

    # 초기 조건
    solver.initialize(alpha_g_init=0.001)
    solver.T_l.set_uniform(300.0)
    solver.T_g.set_uniform(300.0)

    # 경계조건
    # 입구 (하부 중앙)
    solver.set_inlet_bc('inlet_gas', alpha_g_inlet,
                        U_l=[0.0, 0.0], U_g=[0.0, U_g_inlet])
    # 출구 (상부)
    solver.set_outlet_bc('outlet', p_val=0.0)
    # 벽면
    for wall in ['wall_bottom', 'wall_left', 'wall_right']:
        if wall in mesh.boundary_patches:
            solver.set_wall_bc(wall)

    # 솔버 파라미터
    solver.alpha_u = 0.3
    solver.alpha_p = 0.2
    solver.alpha_alpha = 0.3
    solver.tol = 1e-3

    # 비정상 해석
    t_end = 1.0     # 1초
    dt = 0.02

    print(f"  비정상 해석: t_end={t_end}s, dt={dt}s")
    print("  해석 중...")
    result = solver.solve_transient(t_end, dt, report_interval=50)
    print(f"  완료: {result['time_steps']} steps, t={result['final_time']:.3f}s")

    # 결과 분석
    alpha_g = solver.alpha_g.values
    alpha_g_max = float(np.max(alpha_g))
    alpha_g_mean = float(np.mean(alpha_g))

    print(f"  기체 체적분율: max={alpha_g_max:.4f}, mean={alpha_g_mean:.6f}")

    # 셀 좌표
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(mesh.n_cells)])
    y_cells = np.array([mesh.cells[ci].center[1] for ci in range(mesh.n_cells)])

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # 기체 체적분율 분포
    ax = axes[0]
    sc = ax.scatter(x_cells, y_cells, c=alpha_g, cmap='Blues', s=8,
                    vmin=0, vmax=max(alpha_g_max, 0.05))
    plt.colorbar(sc, ax=ax, label=r'$\alpha_g$')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('기체 체적분율 분포')
    ax.set_aspect('equal')

    # 액체 속도장
    ax = axes[1]
    U_l = solver.U_l.values
    u_mag = np.sqrt(U_l[:, 0]**2 + U_l[:, 1]**2)
    sc = ax.scatter(x_cells, y_cells, c=u_mag, cmap='jet', s=8)
    plt.colorbar(sc, ax=ax, label='|U_l| [m/s]')
    # 속도 벡터 (간격을 두고)
    skip = max(1, mesh.n_cells // 200)
    ax.quiver(x_cells[::skip], y_cells[::skip],
              U_l[::skip, 0], U_l[::skip, 1],
              alpha=0.5, scale=2.0)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('액체 속도장')
    ax.set_aspect('equal')

    # 높이별 평균 체적분율
    ax = axes[2]
    n_bins = 20
    y_bins = np.linspace(0, height, n_bins + 1)
    alpha_avg = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_cells >= y_bins[i]) & (y_cells < y_bins[i + 1])
        if np.any(mask):
            alpha_avg[i] = np.mean(alpha_g[mask])

    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    ax.plot(alpha_avg, y_centers, 'b-o', markersize=4)
    ax.set_xlabel(r'평균 $\alpha_g$')
    ax.set_ylabel('y [m]')
    ax.set_title('높이별 평균 기체 체적분율')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case4_bubble_column.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # 잔차 그래프
    if result['residuals']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(result['residuals'])
        ax.set_xlabel('시간 스텝')
        ax.set_ylabel('잔차')
        ax.set_title('기포탑 해석 수렴 이력')
        ax.grid(True, alpha=0.3)
        res_fig_path = os.path.join(figures_dir, 'case4_bubble_column_residuals.png')
        plt.savefig(res_fig_path, dpi=150)
        plt.close()

    result_data = {
        'time_steps': result['time_steps'],
        'final_time': result['final_time'],
        'alpha_g_max': alpha_g_max,
        'alpha_g_mean': alpha_g_mean,
        'figure_path': fig_path,
        'physical_validity': alpha_g_max <= 1.0 and alpha_g_mean < 0.5
    }

    np.savez(os.path.join(results_dir, 'case4_bubble_column.npz'),
             x=x_cells, y=y_cells, alpha_g=alpha_g,
             u_l=solver.U_l.values, u_g=solver.U_g.values)

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case4')
    os.makedirs(case_dir, exist_ok=True)
    params = {
        'W': width, 'H': height, 'nx': nx, 'ny': ny,
        'rho_l': rho_l, 'rho_g': rho_g, 'mu_l': mu_l, 'mu_g': mu_g,
        'd_b': d_b, 't_end': t_end, 'dt': dt,
    }
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {'alpha_g': solver.alpha_g.values}
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case4 export skipped: {e}")

    return result_data


if __name__ == "__main__":
    result = run_case4()
    print(f"\n  결과 요약:")
    print(f"    시간 스텝: {result['time_steps']}")
    print(f"    최대 체적분율: {result['alpha_g_max']:.4f}")
    print(f"    물리적 타당성: {result['physical_validity']}")
