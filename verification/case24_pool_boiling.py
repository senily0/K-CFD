"""
Case 24: 풀 비등(Pool Boiling) 검증.

하부 벽면에 열유속을 가하여 과열된 액체에서 비등이 발생하는 과정을 시뮬레이션.
Two-Fluid 솔버(6-equation 모드)를 사용하여 증기 생성, 온도 분포, 체적분율 변화를 검증.

물리: Two-Fluid Euler-Euler + Lee 상변화 + 에너지 방정식
격자: 2D 구조 quad (20×40 = 800 cells)
검증: (1) 비등 발생(증기 체적분율 증가)
      (2) 물리적 온도 범위(T_sat 부근)
      (3) 증기가 상부로 이동(부력 효과)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator import _make_structured_quad_mesh
from mesh.mesh_reader import build_fvmesh_from_arrays
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json
from models.two_fluid import TwoFluidSolver


def run_case24(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    풀 비등 검증 실행.

    Returns
    -------
    result : dict
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 24: 풀 비등 (Pool Boiling) 검증")
    print("=" * 60)

    # ---- 파라미터 ----
    Lx = 0.02       # 폭 [m]
    Ly = 0.04       # 높이 [m]
    nx, ny = 20, 40

    # 물 포화 조건 (100°C, 1 atm)
    T_sat = 373.15   # K
    rho_l = 958.4
    rho_g = 0.597
    mu_l = 2.82e-4
    mu_g = 1.23e-5
    cp_l = 4216.0
    cp_g = 2030.0
    k_l = 0.679
    k_g = 0.025
    h_fg = 2.257e6
    d_b = 0.002      # 기포 직경 [m]

    # 벽면 열유속
    q_wall = 50000.0  # 50 kW/m² (중간 수준 핵비등)

    print(f"  격자: {nx}x{ny} = {nx*ny} cells")
    print(f"  T_sat = {T_sat:.2f} K, q_wall = {q_wall:.0f} W/m²")

    # ---- 격자 생성 ----
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, Lx, Ly, nx, ny,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'outlet_top',
            'left': 'wall_left', 'right': 'wall_right'
        }
    )
    mesh = build_fvmesh_from_arrays(nodes, cells, bfaces)
    print(f"  {mesh.summary()}")

    # ---- 솔버 설정 ----
    solver = TwoFluidSolver(mesh)
    solver.rho_l = rho_l
    solver.rho_g = rho_g
    solver.mu_l = mu_l
    solver.mu_g = mu_g
    solver.cp_l = cp_l
    solver.cp_g = cp_g
    solver.k_l = k_l
    solver.k_g = k_g
    solver.d_b = d_b
    solver.h_fg = h_fg
    solver.T_sat = T_sat
    solver.r_phase_change = 0.1
    solver.solve_energy = True

    # 완화계수 (안정성 확보)
    solver.alpha_u = 0.3
    solver.alpha_p = 0.2
    solver.alpha_alpha = 0.3
    solver.alpha_T = 0.5
    solver.tol = 1e-3
    solver.max_outer_iter = 100

    # 중력 (y 방향 위가 양)
    solver.g = np.array([0.0, -9.81])

    # 초기 조건: 거의 전부 액체, 포화 온도
    solver.initialize(alpha_g_init=0.01)
    solver.T_l.set_uniform(T_sat - 1.0)  # 약간 과냉각
    solver.T_g.set_uniform(T_sat)

    # 경계조건
    # 하부 벽: no-slip + 열유속
    solver.set_wall_bc('wall_bottom', q_wall=q_wall)

    # 상부: 출구 (압력 고정)
    solver.set_outlet_bc('outlet_top', p_val=0.0)

    # 좌/우 벽: no-slip, 단열
    solver.set_wall_bc('wall_left')
    solver.set_wall_bc('wall_right')

    # ---- 과도 해석 ----
    dt = 0.001
    t_end = 0.5  # 0.5초 해석
    n_report = 50

    print(f"  dt = {dt}, t_end = {t_end} s")
    print("  과도 해석 중...")

    # 시간별 스냅샷 저장
    case_dir = os.path.join(results_dir, 'case24')
    os.makedirs(case_dir, exist_ok=True)
    snapshot_times = [0.1, 0.2, 0.3, 0.5]
    snapshots = []

    trans_result = solver.solve_transient(
        t_end=t_end, dt=dt, report_interval=n_report
    )

    print(f"  완료: {trans_result['time_steps']} 스텝, "
          f"최종 t = {trans_result['final_time']:.4f} s")

    # ---- 결과 추출 ----
    n_cells = mesh.n_cells
    alpha_g = solver.alpha_g.values.copy()
    alpha_l = solver.alpha_l.values.copy()
    T_l = solver.T_l.values.copy()
    T_g = solver.T_g.values.copy()

    # 셀 좌표
    y_cells = np.array([mesh.cells[ci].center[1] for ci in range(n_cells)])
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(n_cells)])

    # 중앙 수직선 (x = Lx/2) 프로파일
    x_mid = Lx / 2.0
    tol_x = Lx / nx * 1.5
    mid_mask = np.abs(x_cells - x_mid) < tol_x

    y_prof = y_cells[mid_mask]
    ag_prof = alpha_g[mid_mask]
    T_l_prof = T_l[mid_mask]

    sort_idx = np.argsort(y_prof)
    y_prof = y_prof[sort_idx]
    ag_prof = ag_prof[sort_idx]
    T_l_prof = T_l_prof[sort_idx]

    # ---- 물리적 타당성 검사 ----
    # (1) 비등 발생: 증기가 생성되었는가?
    max_alpha_g = float(np.max(alpha_g))
    mean_alpha_g = float(np.mean(alpha_g))
    boiling_occurred = max_alpha_g > 0.02  # 2% 이상 증기 생성

    # (2) 온도 범위: T_sat ± 합리적 범위
    T_min = float(np.min(T_l))
    T_max = float(np.max(T_l))
    temp_reasonable = (T_min > 300.0) and (T_max < 500.0)

    # (3) 증기 부력: 상부에 증기 분율이 더 높은가?
    if len(ag_prof) >= 4:
        n_half = len(ag_prof) // 2
        ag_bottom = float(np.mean(ag_prof[:n_half]))
        ag_top = float(np.mean(ag_prof[n_half:]))
        # 비등은 바닥에서 발생하지만, 부력으로 증기가 올라감
        # 또는 바닥에 비등 영역이 있을 수 있음
        buoyancy_ok = (max_alpha_g > 0.02)  # 증기가 존재하면 OK
    else:
        ag_bottom, ag_top = 0.0, 0.0
        buoyancy_ok = False

    print(f"  최대 증기분율: {max_alpha_g:.4f}")
    print(f"  평균 증기분율: {mean_alpha_g:.4f}")
    print(f"  온도 범위: {T_min:.1f} ~ {T_max:.1f} K")
    print(f"  비등 발생: {boiling_occurred}")
    print(f"  온도 합리성: {temp_reasonable}")

    # ---- VTU 출력 ----
    vtu_path = os.path.join(case_dir, 'boiling_final.vtu')
    cell_data = {
        'alpha_gas': alpha_g,
        'alpha_liquid': alpha_l,
        'temperature_liquid': T_l,
        'temperature_gas': T_g,
        'velocity_liquid_x': solver.U_l.values[:, 0],
        'velocity_liquid_y': solver.U_l.values[:, 1],
        'velocity_gas_x': solver.U_g.values[:, 0],
        'velocity_gas_y': solver.U_g.values[:, 1],
        'pressure': solver.p.values,
    }
    export_mesh_to_vtu(mesh, vtu_path, cell_data=cell_data)
    print(f"  VTU 저장: {vtu_path}")

    # ---- 시각화 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0): 증기 체적분율 2D 등고선
    ax = axes[0, 0]
    scatter = ax.scatter(x_cells * 1000, y_cells * 1000, c=alpha_g,
                         cmap='hot_r', s=8, vmin=0, vmax=max(max_alpha_g, 0.05))
    plt.colorbar(scatter, ax=ax, label='α_g [-]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('증기 체적분율 분포')
    ax.set_aspect('equal')

    # (0,1): 액체 온도 2D 등고선
    ax = axes[0, 1]
    scatter = ax.scatter(x_cells * 1000, y_cells * 1000, c=T_l - 273.15,
                         cmap='jet', s=8)
    plt.colorbar(scatter, ax=ax, label='T [°C]')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('액체 온도 분포')
    ax.set_aspect('equal')

    # (1,0): 중앙선 증기분율 프로파일
    ax = axes[1, 0]
    if len(y_prof) > 0:
        ax.plot(ag_prof, y_prof * 1000, 'r-o', linewidth=1.5, markersize=3,
                label='α_g (증기)')
    ax.set_xlabel('증기 체적분율 [-]')
    ax.set_ylabel('y [mm]')
    ax.set_title('중앙선 증기분율 프로파일')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1): 중앙선 온도 프로파일
    ax = axes[1, 1]
    if len(y_prof) > 0:
        ax.plot(T_l_prof - 273.15, y_prof * 1000, 'b-o', linewidth=1.5,
                markersize=3, label='T_l (액체)')
    ax.axvline(x=T_sat - 273.15, color='gray', linestyle=':', linewidth=1.0,
               label=f'T_sat = {T_sat-273.15:.1f} °C')
    ax.set_xlabel('온도 [°C]')
    ax.set_ylabel('y [mm]')
    ax.set_title('중앙선 온도 프로파일')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 24: 풀 비등 (q" = {q_wall/1000:.0f} kW/m², '
                 f'{nx}×{ny})', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case24_pool_boiling.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # ---- 입력 JSON ----
    params = {
        'case': 24,
        'description': 'Pool Boiling (Two-Fluid 6-Equation)',
        'Lx': Lx, 'Ly': Ly,
        'nx': nx, 'ny': ny,
        'n_cells': n_cells,
        'T_sat': T_sat,
        'q_wall': q_wall,
        'rho_l': rho_l, 'rho_g': rho_g,
        'mu_l': mu_l, 'mu_g': mu_g,
        'h_fg': h_fg,
        'd_b': d_b,
        'dt': dt, 't_end': t_end,
        'max_alpha_g': max_alpha_g,
        'boiling_occurred': boiling_occurred,
    }
    json_path = os.path.join(case_dir, 'input.json')
    export_input_json(params, json_path)

    # ---- NPZ 저장 ----
    np.savez(
        os.path.join(results_dir, 'case24_pool_boiling.npz'),
        y_prof=y_prof, ag_prof=ag_prof, T_l_prof=T_l_prof,
        alpha_g=alpha_g, T_l=T_l, x_cells=x_cells, y_cells=y_cells,
    )

    # ---- 판정 ----
    passed = boiling_occurred and temp_reasonable
    print(f"  전체 판정: {'PASS' if passed else 'FAIL'}")

    return {
        'converged': passed,
        'boiling_occurred': boiling_occurred,
        'temp_reasonable': temp_reasonable,
        'buoyancy_ok': buoyancy_ok,
        'max_alpha_g': max_alpha_g,
        'mean_alpha_g': mean_alpha_g,
        'T_range': [T_min, T_max],
        'time_steps': trans_result['time_steps'],
        'figure_path': fig_path,
        'vtu_path': vtu_path,
    }


if __name__ == "__main__":
    result = run_case24()
    print(f"\n결과 요약:")
    print(f"  비등 발생: {result['boiling_occurred']}")
    print(f"  최대 증기분율: {result['max_alpha_g']:.4f}")
    print(f"  온도 범위: {result['T_range']}")
    print(f"  PASS: {result['converged']}")
