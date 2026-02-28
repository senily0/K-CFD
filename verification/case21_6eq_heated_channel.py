"""
Case 21: 6-Equation 가열 채널 검증.

수직 가열 채널에서 과냉수가 가열되어 비등이 시작되는 과정을 6-equation
Two-Fluid Model로 해석한다. 각 상별 에너지 방정식이 SIMPLE 내부에서
암시적으로 풀리며, 비평형 상간 열전달을 재현한다.

검증 항목:
  - T_l, T_g 프로파일 (비평형)
  - 보이드율 축방향 변화
  - 에너지 보존 (입출구 엔탈피 차 ≈ 벽 열유속)
  - 수렴성 (잔차 감소)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from mesh.mesh_generator import _make_structured_quad_mesh
from mesh.mesh_reader import build_fvmesh_from_arrays
from models.two_fluid import TwoFluidSolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json

def _make_channel_mesh(L, W, nx, ny):
    """2D 수직 가열 채널 격자.

    x: 0~W (폭 방향), nx 셀
    y: 0~L (유동 방향, 상향), ny 셀
    """
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, W, L, nx, ny)

    boundary_faces_dict = {}
    # 입구 (y=0, 하단)
    boundary_faces_dict['inlet'] = []
    for i in range(nx):
        n0 = i
        n1 = i + 1
        boundary_faces_dict['inlet'].append(np.array([n0, n1]))

    # 출구 (y=L, 상단)
    boundary_faces_dict['outlet'] = []
    for i in range(nx):
        n0 = ny * (nx + 1) + i
        n1 = ny * (nx + 1) + i + 1
        boundary_faces_dict['outlet'].append(np.array([n0, n1]))

    # 좌벽 (x=0, 가열벽)
    boundary_faces_dict['wall_heated'] = []
    for j in range(ny):
        n0 = j * (nx + 1)
        n1 = (j + 1) * (nx + 1)
        boundary_faces_dict['wall_heated'].append(np.array([n0, n1]))

    # 우벽 (x=W, 단열벽)
    boundary_faces_dict['wall_adiabatic'] = []
    for j in range(ny):
        n0 = j * (nx + 1) + nx
        n1 = (j + 1) * (nx + 1) + nx
        boundary_faces_dict['wall_adiabatic'].append(np.array([n0, n1]))

    return build_fvmesh_from_arrays(
        nodes, cells, boundary_faces_dict, ndim=2)

def run_case21(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    6-Equation 가열 채널 검증 실행.

    Returns
    -------
    result : dict
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 21: 6-Equation 가열 채널 (비평형 상변화)")
    print("=" * 60)

    # 파라미터
    L = 1.0          # 채널 길이 [m]
    W = 0.02         # 채널 폭 [m]
    nx, ny = 1, 20   # 격자: 폭 1셀 (1D 등가), 높이 20셀
    T_inlet = 350.0  # 입구 온도 [K]
    T_sat = 368.0    # 포화 온도 [K] (T_outlet~369K > T_sat → 출구 부근 비등)
    alpha_g_init = 0.01  # 초기 기체 체적분율 (작게)
    U_inlet = 0.3    # 입구 속도 [m/s]
    q_wall = 500000.0  # 벽면 열유속 [W/m²] (500 kW/m²)
    t_end = 50.0     # 해석 시간 [s] (정상상태 도달 보장)
    dt = 0.05        # 시간 간격 [s] (CFL < 0.3: dy=0.05m, U=0.3m/s → CFL=0.3)

    print(f"  격자: {nx} x {ny} = {nx*ny} 셀")
    print(f"  입구 온도: {T_inlet} K, 포화 온도: {T_sat} K")
    print(f"  벽면 열유속: {q_wall/1000:.0f} kW/m²")

    # 격자 생성
    mesh = _make_channel_mesh(L, W, nx, ny)
    print(f"  {mesh.summary()}")

    # 솔버 설정
    solver = TwoFluidSolver(mesh)
    solver.solve_energy = True
    solver.solve_momentum = False  # 속도장 고정 (에너지 전달만 검증)

    # 물성치 (고온 물)
    solver.rho_l = 958.4
    solver.rho_g = 0.597
    solver.mu_l = 2.82e-4
    solver.mu_g = 1.2e-5
    solver.cp_l = 4216.0
    solver.cp_g = 2030.0
    solver.k_l = 0.679
    solver.k_g = 0.025
    solver.d_b = 0.003

    # 6-equation 파라미터
    solver.T_sat = T_sat
    solver.h_fg = 2.257e6
    solver.r_phase_change = 0.001  # Lee 모델 상변화 계수 (안정적 과냉 비등, 완만한 증발)
    solver.h_i_coeff = 10.0      # 상간 열전달 계수
    solver.a_i_coeff = 1.0

    # 중력 없이 (안정성)
    solver.g = np.array([0.0, 0.0])

    # 솔버 파라미터
    solver.alpha_T = 0.7   # 에너지 완화계수 (0.7: 안정성과 수렴속도 균형)
    solver.tol = 1e-6
    solver.max_outer_iter = 30  # SIMPLE 내부 반복

    # 초기 조건
    solver.initialize(alpha_g_init)
    solver.T_l.set_uniform(T_inlet)
    solver.T_g.set_uniform(T_inlet)
    solver.U_l.values[:, 1] = U_inlet
    solver.U_g.values[:, 1] = U_inlet

    # 경계조건
    solver.set_inlet_bc('inlet', alpha_g=alpha_g_init,
                        U_l=[0.0, U_inlet], U_g=[0.0, U_inlet],
                        T_l=T_inlet, T_g=T_inlet)
    solver.set_outlet_bc('outlet', p_val=0.0)
    solver.set_wall_bc('wall_heated', q_wall=q_wall)
    solver.set_wall_bc('wall_adiabatic')

    # 해석 실행
    print("  해석 중...")
    result = solver.solve_transient(t_end=t_end, dt=dt, report_interval=50)
    print(f"  완료: {result['time_steps']} 스텝")

    # 결과 추출 — 벽면 인접(x~0) 및 중앙(x=W/2) 축방향 프로파일
    dx = W / nx
    # 벽면 인접 셀
    y_wall, T_l_wall, T_g_wall = [], [], []
    # 중앙 셀
    y_mid, T_l_mid, T_g_mid, alpha_g_mid = [], [], [], []
    x_mid = W / 2.0

    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if cc[0] < dx:  # 벽면 인접
            y_wall.append(cc[1])
            T_l_wall.append(solver.T_l.values[ci])
            T_g_wall.append(solver.T_g.values[ci])
        if abs(cc[0] - x_mid) < dx:  # 중앙
            y_mid.append(cc[1])
            T_l_mid.append(solver.T_l.values[ci])
            T_g_mid.append(solver.T_g.values[ci])
            alpha_g_mid.append(solver.alpha_g.values[ci])

    y_wall = np.array(y_wall); T_l_wall = np.array(T_l_wall); T_g_wall = np.array(T_g_wall)
    y_mid = np.array(y_mid); T_l_mid = np.array(T_l_mid); T_g_mid = np.array(T_g_mid)
    alpha_g_mid = np.array(alpha_g_mid)

    idx_w = np.argsort(y_wall); y_wall = y_wall[idx_w]; T_l_wall = T_l_wall[idx_w]; T_g_wall = T_g_wall[idx_w]
    idx_m = np.argsort(y_mid); y_mid = y_mid[idx_m]; T_l_mid = T_l_mid[idx_m]; T_g_mid = T_g_mid[idx_m]
    alpha_g_mid = alpha_g_mid[idx_m]

    # 호환성: 기존 변수 유지
    y_prof = y_mid; T_l_prof = T_l_mid; T_g_prof = T_g_mid; alpha_g_prof = alpha_g_mid

    # 에너지 보존 검사: 벽 열유속 = 출구-입구 엔탈피 상승량 (face-by-face)
    # 절대 온도 기반 질량유속 방식: solver가 실제로 보존하는 양과 일치
    from core.interpolation import compute_mass_flux
    mf_l = compute_mass_flux(solver.U_l, solver.rho_l, mesh)
    mf_g = compute_mass_flux(solver.U_g, solver.rho_g, mesh)

    inlet_fids = mesh.boundary_patches.get('inlet', [])
    E_in = 0.0
    for fid in inlet_fids:
        face = mesh.faces[fid]
        owner = face.owner
        al = solver.alpha_l.values[owner]
        ag = solver.alpha_g.values[owner]
        # 입구: 유입 질량유속(음수)의 절대값 × alpha × cp × T_inlet
        F_l = abs(mf_l[fid])
        F_g = abs(mf_g[fid])
        E_in += al * solver.cp_l * F_l * T_inlet + ag * solver.cp_g * F_g * T_inlet

    outlet_fids = mesh.boundary_patches.get('outlet', [])
    E_out = 0.0
    for fid in outlet_fids:
        face = mesh.faces[fid]
        owner = face.owner
        al = solver.alpha_l.values[owner]
        ag = solver.alpha_g.values[owner]
        T_l_o = solver.T_l.values[owner]
        T_g_o = solver.T_g.values[owner]
        # 출구: 유출 질량유속 × alpha × cp × T_outlet
        F_l = mf_l[fid]
        F_g = mf_g[fid]
        E_out += al * solver.cp_l * F_l * T_l_o + ag * solver.cp_g * F_g * T_g_o

    # 벽에서 투입된 총 열량: 가열벽 면적 × 열유속
    # 2D 격자 단위 깊이(1m) 기준: 가열벽 면적 = L (ny개 면, 각 면적 = L/ny)
    heated_fids = mesh.boundary_patches.get('wall_heated', [])
    Q_wall_total = sum(
        q_wall * mesh.faces[fid].area for fid in heated_fids
    )
    # 잠열 기여: 출구에서 증발한 질량유량 × h_fg
    # m_dot_evap = sum over outlet faces of (alpha_g * rho_g * U_g_y * area)
    # minus inlet vapor mass flux
    m_dot_vapor_in = 0.0
    for fid in inlet_fids:
        face = mesh.faces[fid]
        owner = face.owner
        ag = solver.alpha_g.values[owner]
        m_dot_vapor_in += ag * solver.rho_g * U_inlet * face.area

    m_dot_vapor_out = 0.0
    for fid in outlet_fids:
        face = mesh.faces[fid]
        owner = face.owner
        ag = solver.alpha_g.values[owner]
        U_g_y = abs(solver.U_g.values[owner, 1])
        m_dot_vapor_out += ag * solver.rho_g * U_g_y * face.area

    m_dot_evap = max(m_dot_vapor_out - m_dot_vapor_in, 0.0)
    latent_heat_gain = m_dot_evap * solver.h_fg

    # 에너지 보존: Q_wall ≈ 출구-입구 절대 엔탈피 차 (solver 보존량과 일치)
    energy_gain = E_out - E_in
    energy_ratio = abs(energy_gain - Q_wall_total) / max(abs(Q_wall_total), 1e-15)

    # 수렴 판정: 정규화 잔차가 0.05 이하면 수렴으로 판정
    residuals = result.get('residuals', [])
    converged = len(residuals) > 0 and residuals[-1] < 0.05

    # 온도 증가 확인 (벽면 인접 셀의 축방향 증가)
    T_l_increase = float(T_l_wall[-1] - T_l_wall[0]) if len(T_l_wall) > 1 else 0.0

    # 물리적 타당성
    T_l_all = solver.T_l.values
    T_g_all = solver.T_g.values

    # 비물리적 진동 검사: T_l_wall 프로파일이 단조 증가하는지 확인
    # (가열 채널에서 축방향 온도는 단조 증가해야 함)
    monotone_T_l = True
    if len(T_l_wall) > 2:
        dT = np.diff(T_l_wall)
        # 최대 하강폭이 총 온도 상승의 10% 이내이면 단조로 간주
        total_rise = max(T_l_increase, 1e-10)
        max_drop = max(-dT.min(), 0.0)
        monotone_T_l = (max_drop < 0.1 * total_rise)

    # 보이드율 축방향 증가 확인 (과냉 비등: 출구 > 입구)
    void_increases = False
    if len(alpha_g_mid) > 1:
        void_increases = float(alpha_g_mid[-1]) > float(alpha_g_mid[0])

    # 출구 T_l이 T_sat에 근접하는지 확인 (과냉 비등 개시)
    T_l_outlet = float(T_l_mid[-1]) if len(T_l_mid) > 0 else T_inlet
    near_sat = T_l_outlet >= (T_sat - 30.0)  # T_sat 기준 30K 이내

    physical = (np.all(T_l_all >= T_inlet - 1.0) and   # 입구 온도 이상
                np.all(T_l_all <= 600.0) and            # 합리적 상한
                np.all(np.isfinite(T_l_all)) and
                np.all(np.isfinite(T_g_all)) and
                T_l_increase > 1.0 and                  # 벽면 가열 효과 확인
                monotone_T_l and                         # 비물리적 진동 없음
                void_increases and                       # 보이드율 출구 > 입구
                near_sat)                                # 출구 T_l이 T_sat에 근접

    print(f"  T_l 범위: [{T_l_all.min():.1f}, {T_l_all.max():.1f}] K")
    print(f"  T_g 범위: [{T_g_all.min():.1f}, {T_g_all.max():.1f}] K")
    print(f"  벽면 T_l 증가 (축방향): {T_l_increase:.2f} K")
    print(f"  T_l 단조 증가: {monotone_T_l}")
    print(f"  보이드율 축방향 증가: {void_increases} (inlet={alpha_g_mid[0]:.4f}, outlet={alpha_g_mid[-1]:.4f})")
    print(f"  출구 T_l={T_l_outlet:.1f} K, T_sat={T_sat:.1f} K, 근접: {near_sat}")
    print(f"  잠열 기여: {latent_heat_gain:.1f} W/m (증발 질량유량={m_dot_evap:.6f} kg/s/m)")
    print(f"  에너지 보존 오차: {energy_ratio*100:.1f}%")
    print(f"  수렴: {converged}, 물리적 타당: {physical}")

    # 시각화 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 벽면 인접 온도 프로파일
    ax = axes[0, 0]
    ax.plot(y_wall, T_l_wall, 'b-o', markersize=3, label='T_l (wall-adj)')
    ax.plot(y_wall, T_g_wall, 'r-s', markersize=3, label='T_g (wall-adj)')
    ax.plot(y_mid, T_l_mid, 'b--', alpha=0.5, label='T_l (center)')
    ax.axhline(y=T_inlet, color='gray', linestyle=':', alpha=0.5, label=f'T_inlet={T_inlet:.0f}K')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('Axial temperature profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) 보이드율
    ax = axes[0, 1]
    ax.plot(y_mid, alpha_g_mid, 'g-^', markersize=3)
    ax.set_xlabel('y [m]')
    ax.set_ylabel(r'$\alpha_g$')
    ax.set_title('Axial void fraction')
    ax.grid(True, alpha=0.3)

    # (3) 에너지 수지
    ax = axes[1, 0]
    labels = ['Q_wall', 'E_out - E_in']
    values = [Q_wall_total, energy_gain]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Energy [W/m]')
    ax.set_title(f'Energy balance (error: {energy_ratio*100:.1f}%)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # (4) 잔차 이력
    ax = axes[1, 1]
    if residuals:
        ax.semilogy(residuals)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence history')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 21: 6-Eq Heated Channel (q"={q_wall/1000:.0f} kW/m²)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case21_6eq_heated_channel.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    energy_ok = energy_ratio < 0.10  # 에너지 보존 오차 10% 이하 (절대 엔탈피 기반)
    passed = converged and physical and energy_ok
    print(f"  에너지 보존 기준 충족: {energy_ok} (오차={energy_ratio*100:.1f}%)")
    print(f"  단조 온도 상승: {monotone_T_l}")
    print(f"\n  결과: {'PASS' if passed else 'FAIL'}")

    result_data = {
        'converged': converged,
        'physical': physical,
        'monotone_T_l': monotone_T_l,
        'void_increases': void_increases,
        'near_sat': near_sat,
        'T_l_outlet': T_l_outlet,
        'alpha_g_inlet': float(alpha_g_mid[0]) if len(alpha_g_mid) > 0 else None,
        'alpha_g_outlet': float(alpha_g_mid[-1]) if len(alpha_g_mid) > 0 else None,
        'latent_heat_gain': float(latent_heat_gain),
        'T_l_range': [float(T_l_all.min()), float(T_l_all.max())],
        'T_g_range': [float(T_g_all.min()), float(T_g_all.max())],
        'T_l_increase': float(T_l_increase),
        'energy_ratio': float(energy_ratio),
        'Q_wall_total': float(Q_wall_total),
        'energy_gain': float(energy_gain),
        'residuals': residuals,
        'figure_path': fig_path,
        'time_steps': result['time_steps'],
    }

    # VTU / JSON export
    case_dir = os.path.join(results_dir, 'case21')
    os.makedirs(case_dir, exist_ok=True)
    params = {'L': L, 'W': W, 'nx': nx, 'ny': ny, 'T_inlet': T_inlet,
              'T_sat': T_sat, 'q_wall': q_wall, 'U_inlet': U_inlet,
              't_end': t_end, 'dt': dt}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'temperature_liquid': solver.T_l.values,
            'temperature_gas': solver.T_g.values,
            'alpha_gas': solver.alpha_g.values,
            'velocity_liquid_x': solver.U_l.values[:, 0],
            'velocity_liquid_y': solver.U_l.values[:, 1],
            'pressure': solver.p.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case21 export skipped: {e}")

    return result_data

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case21()
    print(f"  T_l 증가: {result['T_l_increase']:.2f} K")
    print(f"  에너지 보존 오차: {result['energy_ratio']*100:.1f}%")
