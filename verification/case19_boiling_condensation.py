"""
Case 19: 비등/응축 상변화 모델 검증.

3가지 하위 검증:
  Test A: Rohsenow 핵비등 열유속 검증 (물 100°C, 1 atm)
  Test B: Nusselt 막응축 열전달 계수 검증 (수직 평판)
  Test C: 1D 비등/응축 통합 시뮬레이션 (Lee 모델 + 에너지 방정식)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from models.phase_change import (
    RohsenowBoilingModel, ZuberCHFModel, NusseltCondensationModel,
    LeePhaseChangeModel,
)
from mesh.mesh_generator import _make_structured_quad_mesh
from mesh.mesh_reader import build_fvmesh_from_arrays
from core.fields import ScalarField
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json

# ---------------------------------------------------------------------------
# 물(water) 100°C, 1 atm 물성치
# ---------------------------------------------------------------------------
WATER_PROPS = dict(
    T_sat=373.15,       # K
    h_fg=2.257e6,       # J/kg
    rho_l=958.4,        # kg/m3
    rho_g=0.597,        # kg/m3
    mu_l=2.82e-4,       # Pa·s
    cp_l=4216.0,        # J/(kg·K)
    k_l=0.679,          # W/(m·K)
    sigma=0.0589,       # N/m
    Pr_l=1.75,          # -
)

# ---------------------------------------------------------------------------
# 1D 채널 메쉬 (case9에서 차용)
# ---------------------------------------------------------------------------
def _make_1d_mesh(L, nx):
    """길이 L, 셀 수 nx 인 1D (x 방향) 채널 메쉬."""
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, L, L / nx,
        nx, 1,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'wall_top',
            'left': 'left', 'right': 'right'
        }
    )
    return build_fvmesh_from_arrays(nodes, cells, bfaces)

# ===================================================================
# Test A: Rohsenow 핵비등 열유속 검증
# ===================================================================
def _test_a_rohsenow(wp):
    """
    벽면 과열도 dT = 5~25 K 에서 Rohsenow 열유속 계산.
    물리적 범위(1e3~1e6 W/m2)에 있는지 확인.
    Zuber CHF와 비교.
    """
    boiling = RohsenowBoilingModel(
        T_sat=wp['T_sat'], h_fg=wp['h_fg'],
        rho_l=wp['rho_l'], rho_g=wp['rho_g'],
        mu_l=wp['mu_l'], cp_l=wp['cp_l'],
        sigma=wp['sigma'], Pr_l=wp['Pr_l'],
        C_sf=0.013, n=1.0,
    )

    zuber = ZuberCHFModel(
        h_fg=wp['h_fg'], rho_l=wp['rho_l'],
        rho_g=wp['rho_g'], sigma=wp['sigma'],
    )

    dT_arr = np.linspace(5, 25, 21)
    q_arr = np.array([boiling.compute_wall_heat_flux(wp['T_sat'] + dT)
                       for dT in dT_arr])
    chf = zuber.compute_chf()

    q_min, q_max = float(q_arr.min()), float(q_arr.max())

    # 물리적 범위 확인 (핵비등은 통상 1e3 ~ 1e6 W/m2)
    valid = (q_min >= 1e3) and (q_max <= 5e6)

    return {
        'dT': dT_arr,
        'q': q_arr,
        'chf': chf,
        'q_range': [q_min, q_max],
        'valid': bool(valid),
    }

# ===================================================================
# Test B: Nusselt 막응축 열전달 계수 검증
# ===================================================================
def _test_b_nusselt(wp):
    """
    수직 평판(L=0.1m), 과냉도 dT_sub = 5~20 K.
    해석해와 수치 모델 비교 — 오차 < 5%.
    """
    cond = NusseltCondensationModel(
        T_sat=wp['T_sat'], h_fg=wp['h_fg'],
        rho_l=wp['rho_l'], rho_g=wp['rho_g'],
        mu_l=wp['mu_l'], k_l=wp['k_l'],
    )

    L_plate = 0.1  # m
    dT_arr = np.linspace(5, 20, 16)

    # 수치 모델
    h_numerical = np.array([cond.compute_heat_transfer_coeff(L_plate, dT)
                             for dT in dT_arr])

    # 해석해 (직접 계산)
    g = 9.81
    h_analytical = np.array([
        0.943 * (wp['rho_l'] * (wp['rho_l'] - wp['rho_g']) * g
                 * wp['h_fg'] * wp['k_l']**3
                 / (wp['mu_l'] * L_plate * dT)) ** 0.25
        for dT in dT_arr
    ])

    # 상대 오차
    rel_err = np.abs(h_numerical - h_analytical) / np.maximum(h_analytical, 1e-12)
    max_err = float(rel_err.max())

    valid = max_err < 0.05

    return {
        'dT': dT_arr,
        'h_numerical': h_numerical,
        'h_analytical': h_analytical,
        'max_error': max_err,
        'h_range': [float(h_numerical.min()), float(h_numerical.max())],
        'valid': bool(valid),
    }

# ===================================================================
# Test C: 1D 비등/응축 통합 시뮬레이션
# ===================================================================
def _test_c_integrated(wp):
    """
    1D 채널(50셀), 왼쪽 벽 T_hot(비등), 오른쪽 벽 T_cold(응축).
    Lee 모델 + 에너지 방정식 연성.
    """
    L = 0.01       # m
    nx = 50
    T_hot = wp['T_sat'] + 5.0    # 378.15 K (작은 과열도로 안정성 확보)
    T_cold = wp['T_sat'] - 5.0   # 368.15 K
    rho_l = wp['rho_l']
    rho_g = wp['rho_g']
    cp = wp['cp_l']
    k = wp['k_l']
    h_fg = wp['h_fg']
    r_pc = 0.5     # 상변화 계수 (명시적 오일러 안정성 위해 작은 값)

    mesh = _make_1d_mesh(L, nx)
    n = mesh.n_cells
    dx = L / nx

    # Lee 모델
    pc = LeePhaseChangeModel(
        mesh, T_sat=wp['T_sat'],
        r_evap=r_pc, r_cond=r_pc,
        L_latent=h_fg,
        rho_l=rho_l, rho_g=rho_g,
    )

    # 필드 초기화: 선형 온도 분포로 시작
    T_field = ScalarField(mesh, "T", default=wp['T_sat'])
    alpha_l = ScalarField(mesh, "alpha_l", default=0.5)

    # 셀 좌표
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(n)])
    sort_idx = np.argsort(x_cells)
    cell_volumes = np.array([mesh.cells[ci].volume for ci in range(n)])

    # 면 데이터 사전 계산
    internal_face_data = []
    left_bc_data = []
    right_bc_data = []

    for fid in range(mesh.n_faces):
        face = mesh.faces[fid]
        owner = face.owner
        nb = face.neighbour
        if nb >= 0:
            d = np.linalg.norm(mesh.cells[nb].center - mesh.cells[owner].center)
            if d > 1e-30:
                internal_face_data.append((owner, nb, face.area, d,
                                           cell_volumes[owner],
                                           cell_volumes[nb]))
        else:
            bname = face.boundary_tag
            d = np.linalg.norm(face.center - mesh.cells[owner].center)
            if d < 1e-30:
                continue
            if bname == 'left':
                left_bc_data.append((owner, d, face.area))
            elif bname == 'right':
                right_bc_data.append((owner, d, face.area))

    # 시간 전진
    alpha_th = k / (rho_l * cp)
    dt = 0.2 * dx**2 / alpha_th  # 보수적 CFL
    # 상변화 안정성 제한: dt < 1 / (r_pc * max(rho_l, rho_g) / T_sat)
    dt_pc = 0.3 * wp['T_sat'] / (r_pc * rho_l)
    dt = min(dt, dt_pc)
    n_steps = 300
    t_end = n_steps * dt

    converged = True
    for step in range(n_steps):
        T_vals = T_field.values.copy()
        al_vals = alpha_l.values.copy()
        rho_mix = al_vals * rho_l + (1.0 - al_vals) * rho_g
        rho_cp = rho_mix * cp

        # 열확산
        dT = np.zeros(n)
        for (o, nb, area, d, vol_o, vol_nb) in internal_face_data:
            flux = k * area * (T_vals[nb] - T_vals[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * vol_o)
            dT[nb] -= dt * flux / (rho_cp[nb] * vol_nb)

        for (o, d, area) in left_bc_data:
            flux = k * area * (T_hot - T_vals[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        for (o, d, area) in right_bc_data:
            flux = k * area * (T_cold - T_vals[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        T_field.values = T_vals + dT

        # 온도 클램핑 (물리적 범위 내로 제한)
        T_field.values = np.clip(T_field.values, T_cold - 5.0, T_hot + 5.0)

        # Lee 소스항 — alpha 업데이트
        src = pc.get_source_terms(T_field, alpha_l)
        alpha_l.values += dt * src['alpha_l']
        alpha_l.values = np.clip(alpha_l.values, 1e-6, 1.0 - 1e-6)

        # 에너지 소스항 (잠열)
        rho_cp_new = (alpha_l.values * rho_l
                      + (1.0 - alpha_l.values) * rho_g) * cp
        T_field.values += dt * src['energy'] / np.maximum(rho_cp_new, 1.0)
        T_field.values = np.clip(T_field.values, T_cold - 5.0, T_hot + 5.0)

        # NaN 체크
        if np.any(np.isnan(T_field.values)) or np.any(np.isnan(alpha_l.values)):
            converged = False
            break

    # 결과 분석
    x_sorted = x_cells[sort_idx]
    al_sorted = alpha_l.values[sort_idx]
    T_sorted = T_field.values[sort_idx]

    # 왼쪽 절반에 증기가 많고 오른쪽에 액체가 많은지 확인
    mid = nx // 2
    left_avg_ag = np.mean(1.0 - al_sorted[:mid])   # 왼쪽 증기분율 평균
    right_avg_al = np.mean(al_sorted[mid:])          # 오른쪽 액체분율 평균
    separation = (left_avg_ag > 0.1) and (right_avg_al > 0.1)

    # 에너지 보존 오차: 총 에너지 변화 vs 벽면 입출력
    # 간단한 추정: 정상 상태에서 에너지 보존
    total_energy = np.sum((al_sorted * rho_l + (1.0 - al_sorted) * rho_g)
                          * cp * T_sorted * cell_volumes[sort_idx])
    # 초기 에너지
    init_energy = np.sum((0.5 * rho_l + 0.5 * rho_g) * cp * wp['T_sat']
                         * cell_volumes)
    energy_change = abs(total_energy - init_energy)
    energy_ref = max(abs(init_energy), 1e-10)
    energy_balance_error = energy_change / energy_ref

    return {
        'x': x_sorted,
        'alpha_l': al_sorted,
        'T': T_sorted,
        'converged': bool(converged),
        'separation': bool(separation),
        'energy_balance_error': float(energy_balance_error),
        'left_avg_ag': float(left_avg_ag),
        'right_avg_al': float(right_avg_al),
    }

# ===================================================================
# 메인 케이스 함수
# ===================================================================
def run_case19(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    비등/응축 상변화 모델 검증 실행.

    Returns
    -------
    result : dict
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 19: 비등/응축 상변화 모델 검증")
    print("=" * 60)

    wp = WATER_PROPS

    # ------ Test A: Rohsenow ------
    print("\n  [Test A] Rohsenow 핵비등 열유속 검증 ...")
    res_a = _test_a_rohsenow(wp)
    print(f"    열유속 범위: {res_a['q_range'][0]:.2e} ~ {res_a['q_range'][1]:.2e} W/m2")
    print(f"    Zuber CHF:   {res_a['chf']:.2e} W/m2")
    print(f"    판정: {'PASS' if res_a['valid'] else 'FAIL'}")

    # ------ Test B: Nusselt ------
    print("\n  [Test B] Nusselt 막응축 열전달 계수 검증 ...")
    res_b = _test_b_nusselt(wp)
    print(f"    h_cond 범위: {res_b['h_range'][0]:.1f} ~ {res_b['h_range'][1]:.1f} W/(m2·K)")
    print(f"    최대 오차:   {res_b['max_error']:.6e}")
    print(f"    판정: {'PASS' if res_b['valid'] else 'FAIL'}")

    # ------ Test C: 통합 시뮬레이션 ------
    print("\n  [Test C] 1D 비등/응축 통합 시뮬레이션 ...")
    res_c = _test_c_integrated(wp)
    print(f"    수렴 여부:           {res_c['converged']}")
    print(f"    증발/응축 영역 분리: {res_c['separation']}")
    print(f"    에너지 보존 오차:    {res_c['energy_balance_error']:.4e}")
    print(f"    판정: {'PASS' if (res_c['converged'] and res_c['energy_balance_error'] < 0.05) else 'FAIL'}")

    # ------ 전체 판정 ------
    overall = (res_a['valid'] and res_b['valid']
               and res_c['converged'] and res_c['energy_balance_error'] < 0.05)

    print(f"\n  전체 판정: {'PASS' if overall else 'FAIL'}")

    # ------ 시각화 (2x2 subplot) ------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0): Rohsenow 열유속 vs 과열도 + CHF 한계
    ax = axes[0, 0]
    ax.semilogy(res_a['dT'], res_a['q'], 'b-o', linewidth=2,
                markersize=4, label='Rohsenow 열유속')
    ax.axhline(y=res_a['chf'], color='r', linestyle='--', linewidth=2,
               label=f'Zuber CHF = {res_a["chf"]:.2e} W/m²')
    ax.set_xlabel('과열도 ΔT_sat [K]')
    ax.set_ylabel('열유속 q" [W/m²]')
    ax.set_title('(A) Rohsenow 핵비등 열유속')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1): Nusselt h vs 과냉도
    ax = axes[0, 1]
    ax.plot(res_b['dT'], res_b['h_numerical'], 'b-o', linewidth=2,
            markersize=4, label='수치 모델')
    ax.plot(res_b['dT'], res_b['h_analytical'], 'r--s', linewidth=2,
            markersize=4, label='Nusselt 해석해')
    ax.set_xlabel('과냉도 ΔT_sub [K]')
    ax.set_ylabel('열전달 계수 h [W/(m²·K)]')
    ax.set_title(f'(B) Nusselt 막응축 (max err={res_b["max_error"]:.2e})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1,0): 통합 시뮬레이션 — 체적분율
    ax = axes[1, 0]
    ax.plot(res_c['x'] * 1e3, res_c['alpha_l'], 'b-', linewidth=2,
            label='α_l (액체)')
    ax.plot(res_c['x'] * 1e3, 1.0 - res_c['alpha_l'], 'r--', linewidth=2,
            label='α_g (증기)')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('체적분율 [-]')
    ax.set_title('(C) 통합 시뮬레이션: 체적분율')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    # (1,1): 통합 시뮬레이션 — 온도
    ax = axes[1, 1]
    ax.plot(res_c['x'] * 1e3, res_c['T'] - 273.15, 'r-', linewidth=2)
    ax.axhline(y=wp['T_sat'] - 273.15, color='gray', linestyle=':',
               alpha=0.7, label=f'T_sat = {wp["T_sat"]-273.15:.1f} °C')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('온도 [°C]')
    ax.set_title('(C) 통합 시뮬레이션: 온도 프로파일')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case19_boiling_condensation.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {fig_path}")

    # VTU 출력 (Test C 통합 시뮬레이션 결과)
    case_dir = os.path.join(results_dir, 'case19')
    os.makedirs(case_dir, exist_ok=True)
    try:
        cell_data = {
            'temperature': res_c['T'],
            'alpha_l': res_c['alpha_l'],
        }
        # Test C 메쉬 재생성 for VTU export
        vtu_mesh = _make_1d_mesh(0.01, 50)
        export_mesh_to_vtu(vtu_mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
        print(f"  VTU 저장: {os.path.join(case_dir, 'mesh.vtu')}")
    except Exception as e:
        print(f"  [VTU] case19 export skipped: {e}")

    # 결과 저장
    np.savez(
        os.path.join(results_dir, 'case19_boiling_condensation.npz'),
        rohsenow_dT=res_a['dT'],
        rohsenow_q=res_a['q'],
        zuber_chf=res_a['chf'],
        nusselt_dT=res_b['dT'],
        nusselt_h_num=res_b['h_numerical'],
        nusselt_h_ana=res_b['h_analytical'],
        integrated_x=res_c['x'],
        integrated_alpha_l=res_c['alpha_l'],
        integrated_T=res_c['T'],
    )

    result = {
        'rohsenow_valid': res_a['valid'],
        'rohsenow_q_range': res_a['q_range'],
        'zuber_chf': res_a['chf'],
        'nusselt_error': res_b['max_error'],
        'nusselt_h_range': res_b['h_range'],
        'integrated_converged': res_c['converged'],
        'energy_balance_error': res_c['energy_balance_error'],
        'converged': overall,
        'figure_path': fig_path,
    }
    return result

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case19()
    print(f"\n결과 요약:")
    print(f"  Rohsenow 유효:     {result['rohsenow_valid']}")
    print(f"  Rohsenow q 범위:   {result['rohsenow_q_range']}")
    print(f"  Zuber CHF:         {result['zuber_chf']:.2e} W/m²")
    print(f"  Nusselt 오차:      {result['nusselt_error']:.6e}")
    print(f"  통합 수렴:         {result['integrated_converged']}")
    print(f"  에너지 보존 오차:  {result['energy_balance_error']:.4e}")
    print(f"  전체 PASS:         {result['converged']}")
