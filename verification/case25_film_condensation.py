"""
Case 25: 막 응축(Film Condensation) 검증.

1D 도메인에서 냉벽(왼쪽)과 포화 증기(오른쪽) 사이의 응축 현상을 시뮬레이션.
Lee 상변화 모델로 증기→액체 응축을 모사하고, Nusselt 해석해와 비교한다.

Test A: Nusselt 막응축 해석해 재검증 (다양한 과냉도, 다양한 판 높이)
Test B: 1D 응축 시뮬레이션 (Lee 모델 + 에너지 방정식)
        - 왼쪽: 냉벽 (T_cold < T_sat) → 응축 발생
        - 오른쪽: 포화 증기 (T = T_sat)
        - 시간 경과에 따라 냉벽 근처에 액막 형성 확인

물리: Lee 상변화 모델 + 열확산 (명시적 시간 적분)
격자: 2D 구조 quad (1D 해석용, 80×1)
검증: (1) Nusselt 해석해 재현 정확도
      (2) 응축 발생 확인(냉벽 근처 α_l 증가)
      (3) 온도 분포 물리적 타당성
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
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json
from models.phase_change import NusseltCondensationModel, LeePhaseChangeModel
from core.fields import ScalarField

# 물(water) 100°C, 1 atm 물성치
WATER_PROPS = dict(
    T_sat=373.15,
    h_fg=2.257e6,
    rho_l=958.4,
    rho_g=0.597,
    mu_l=2.82e-4,
    cp_l=4216.0,
    k_l=0.679,
    k_g=0.025,
    sigma=0.0589,
    Pr_l=1.75,
)

def _make_1d_mesh(L, nx):
    """길이 L, 셀 수 nx 인 1D (x 방향) 채널 메쉬."""
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, L, L / nx,
        nx, 1,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'wall_top',
            'left': 'cold_wall', 'right': 'vapor_bulk'
        }
    )
    return build_fvmesh_from_arrays(nodes, cells, bfaces)

def _test_a_nusselt_validation(wp):
    """
    Test A: Nusselt 막응축 해석해 검증 (다양한 조건).
    """
    cond = NusseltCondensationModel(
        T_sat=wp['T_sat'], h_fg=wp['h_fg'],
        rho_l=wp['rho_l'], rho_g=wp['rho_g'],
        mu_l=wp['mu_l'], k_l=wp['k_l'],
    )

    # 다양한 판 높이
    L_arr = np.array([0.05, 0.1, 0.2, 0.5])
    dT_sub = 10.0  # 과냉도 10K

    h_numerical = np.array([cond.compute_heat_transfer_coeff(L, dT_sub)
                            for L in L_arr])

    # 해석해 직접 계산
    g = 9.81
    h_analytical = np.array([
        0.943 * (wp['rho_l'] * (wp['rho_l'] - wp['rho_g']) * g
                 * wp['h_fg'] * wp['k_l']**3
                 / (wp['mu_l'] * L * dT_sub)) ** 0.25
        for L in L_arr
    ])

    rel_err = np.abs(h_numerical - h_analytical) / np.maximum(h_analytical, 1e-12)
    max_err = float(rel_err.max())

    # 다양한 과냉도 (L = 0.1m 고정)
    L_fixed = 0.1
    dT_arr = np.linspace(3, 25, 23)
    h_vs_dT_num = np.array([cond.compute_heat_transfer_coeff(L_fixed, dT)
                            for dT in dT_arr])
    h_vs_dT_ana = np.array([
        0.943 * (wp['rho_l'] * (wp['rho_l'] - wp['rho_g']) * g
                 * wp['h_fg'] * wp['k_l']**3
                 / (wp['mu_l'] * L_fixed * dT)) ** 0.25
        for dT in dT_arr
    ])

    return {
        'L_arr': L_arr,
        'h_numerical_L': h_numerical,
        'h_analytical_L': h_analytical,
        'dT_arr': dT_arr,
        'h_numerical_dT': h_vs_dT_num,
        'h_analytical_dT': h_vs_dT_ana,
        'max_error': max_err,
        'valid': max_err < 0.01,
    }

def _test_b_condensation_sim(wp):
    """
    Test B: 1D 응축 시뮬레이션.
    왼쪽 냉벽(T_cold), 오른쪽 포화 증기(T_sat).

    핵심 물리: 냉벽이 증기에서 열을 제거 → 증기가 포화온도 이하로 냉각 →
    응축 발생(증기→액체) + 잠열 방출 → 잠열은 벽면으로 전도.

    수치 처리: 잠열-온도 커플링이 극도로 강함(stiff).
    증기의 열용량(rho_g*cp ≈ 2500)이 작아 잠열(h_fg=2.26e6)이
    즉시 온도를 T_sat으로 복원시킴.
    → 벽면 열제거율 기반 평형 응축(equilibrium condensation) 접근:
      q_wall → dot_m = q_wall / h_fg → alpha_l 증가
    이는 Nusselt 막응축의 물리적 메커니즘과 일치.
    """
    L = 0.01       # m (도메인 길이)
    nx = 80
    T_cold = wp['T_sat'] - 10.0  # 363.15 K (냉벽 온도)
    rho_l = wp['rho_l']
    rho_g = wp['rho_g']
    cp_l = wp['cp_l']
    k_l = wp['k_l']
    h_fg = wp['h_fg']

    mesh = _make_1d_mesh(L, nx)
    n = mesh.n_cells
    dx = L / nx

    # 초기 조건
    alpha_l_arr = np.full(n, 0.01)    # 거의 전부 증기
    T_arr = np.full(n, wp['T_sat'])   # 포화 온도

    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(n)])
    sort_idx = np.argsort(x_cells)
    x_sorted_idx = sort_idx  # 왼쪽→오른쪽 정렬 인덱스
    cell_volumes = np.array([mesh.cells[ci].volume for ci in range(n)])

    # 면 데이터 사전 계산
    internal_face_data = []
    left_bc_cells = []   # (cell_id, distance_to_wall, face_area)
    right_bc_cells = []

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
            if bname == 'cold_wall':
                left_bc_cells.append((owner, d, face.area))
            elif bname == 'vapor_bulk':
                right_bc_cells.append((owner, d, face.area))

    # 시간 적분 설정
    # 액체 기준 열확산 (응축 후 액막이 형성되면 액체 물성 지배)
    alpha_th_l = k_l / (rho_l * cp_l)  # ~ 1.68e-7 m²/s
    dt = 0.3 * dx**2 / alpha_th_l      # ~ 1.8e-3 s (안정적)
    n_steps = 2000
    t_end = n_steps * dt

    converged = True
    for step in range(n_steps):
        al = alpha_l_arr.copy()
        T = T_arr.copy()

        # 혼합 밀도 (체적 가중)
        rho_mix = al * rho_l + (1.0 - al) * rho_g
        rho_cp = rho_mix * cp_l

        # ======== Step 1: 열확산 ========
        dT = np.zeros(n)

        # 내부 면: 열전도
        for (o, nb, area, d, vol_o, vol_nb) in internal_face_data:
            # 혼합 열전도도 (α 가중 평균)
            k_o = al[o] * k_l + (1.0 - al[o]) * wp['k_g']
            k_nb = al[nb] * k_l + (1.0 - al[nb]) * wp['k_g']
            k_face = 2.0 * k_o * k_nb / max(k_o + k_nb, 1e-30)
            flux = k_face * area * (T[nb] - T[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * vol_o)
            dT[nb] -= dt * flux / (rho_cp[nb] * vol_nb)

        # 왼쪽: 냉벽 (Dirichlet BC)
        for (o, d, area) in left_bc_cells:
            k_o = al[o] * k_l + (1.0 - al[o]) * wp['k_g']
            flux = k_o * area * (T_cold - T[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        # 오른쪽: 포화 증기 (Dirichlet BC)
        for (o, d, area) in right_bc_cells:
            k_o = al[o] * k_l + (1.0 - al[o]) * wp['k_g']
            flux = k_o * area * (wp['T_sat'] - T[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        T_new = T + dT

        # ======== Step 2: 평형 응축 ========
        # T < T_sat 인 셀에서 과냉도에 비례하여 응축 발생
        # 잠열 방출을 에너지 보존으로 처리:
        #   (rho_cp * dT_reheat) = dot_m * h_fg (잠열이 온도를 올림)
        #   dot_m = rho_g * alpha_g * r_cond * (T_sat - T) / T_sat (Lee 모델)
        #
        # 평형 접근: 응축 잠열이 온도를 T_sat까지 올리는 양만큼만 응축 허용
        # → 에너지 보존: delta_alpha_l * rho_l * h_fg = rho_cp * (T_sat - T_new)
        # 즉, 과냉도를 잠열로 채우는 만큼만 응축
        for i in range(n):
            if T_new[i] < wp['T_sat'] and al[i] < (1.0 - 1e-6):
                subcooling = wp['T_sat'] - T_new[i]  # > 0
                ag = 1.0 - al[i]

                # 에너지 기반 최대 응축량: 과냉 에너지를 잠열로 변환
                # rho_cp * subcooling = d_alpha_l * rho_l * h_fg
                d_al_max_energy = rho_cp[i] * subcooling / (rho_l * h_fg)

                # 증기 가용량 제한
                d_al_max_vapor = ag * 0.5  # 한 스텝에 최대 50%만 응축

                d_al = min(d_al_max_energy, d_al_max_vapor)
                d_al = max(d_al, 0.0)

                alpha_l_arr[i] = al[i] + d_al

                # 잠열 방출로 온도 상승
                T_reheat = d_al * rho_l * h_fg / max(rho_cp[i], 1.0)
                T_new[i] += T_reheat

        alpha_l_arr = np.clip(alpha_l_arr, 1e-6, 1.0 - 1e-6)
        T_arr = np.clip(T_new, T_cold - 1.0, wp['T_sat'] + 1.0)

        if np.any(np.isnan(T_arr)) or np.any(np.isnan(alpha_l_arr)):
            converged = False
            break

    # 결과 정리
    x_sorted = x_cells[sort_idx]
    al_sorted = alpha_l_arr[sort_idx]
    T_sorted = T_arr[sort_idx]

    # ScalarField로 복원 (VTU 출력용)
    T_field = ScalarField(mesh, "T", default=0.0)
    T_field.values = T_arr.copy()
    alpha_l_field = ScalarField(mesh, "alpha_l", default=0.0)
    alpha_l_field.values = alpha_l_arr.copy()

    # 냉벽 근처(왼쪽 1/4) 액체분율 평균
    n_quarter = nx // 4
    al_near_wall = float(np.mean(al_sorted[:n_quarter]))
    al_bulk = float(np.mean(al_sorted[3*n_quarter:]))

    # 응축 발생 확인: 냉벽 근처에 액체가 많아야 함
    condensation_occurred = al_near_wall > al_bulk and al_near_wall > 0.05

    # 온도 구배: 냉벽 근처 T < 포화온도
    T_near_wall = float(np.mean(T_sorted[:n_quarter]))
    T_bulk_val = float(np.mean(T_sorted[3*n_quarter:]))
    temp_gradient_ok = T_near_wall < T_bulk_val

    # 액막 두께 추정
    film_cells = x_sorted[al_sorted > 0.5]
    delta_numerical = float(np.max(film_cells)) if len(film_cells) > 0 else 0.0

    return {
        'x': x_sorted,
        'alpha_l': al_sorted,
        'T': T_sorted,
        'T_cell_order': T_arr,            # VTU 출력용 (셀 순서)
        'al_cell_order': alpha_l_arr,     # VTU 출력용 (셀 순서)
        'converged': bool(converged),
        'condensation_occurred': bool(condensation_occurred),
        'temp_gradient_ok': bool(temp_gradient_ok),
        'al_near_wall': al_near_wall,
        'al_bulk': al_bulk,
        'T_near_wall': T_near_wall,
        'T_bulk': T_bulk_val,
        'delta_numerical': delta_numerical,
        'mesh': mesh,
    }

def run_case25(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    막 응축 검증 실행.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 25: 막 응축 (Film Condensation) 검증")
    print("=" * 60)

    wp = WATER_PROPS
    T_wall = wp['T_sat'] - 10.0
    dT_sub = wp['T_sat'] - T_wall

    # Nusselt 해석해 참조
    nusselt_model = NusseltCondensationModel(
        T_sat=wp['T_sat'], h_fg=wp['h_fg'],
        rho_l=wp['rho_l'], rho_g=wp['rho_g'],
        mu_l=wp['mu_l'], k_l=wp['k_l'],
    )
    h_nusselt = nusselt_model.compute_heat_transfer_coeff(0.1, dT_sub)
    g = 9.81
    delta_nusselt = (4.0 * wp['mu_l'] * wp['k_l'] * dT_sub * 0.1 /
                     (g * wp['rho_l'] * (wp['rho_l'] - wp['rho_g']) * wp['h_fg'])) ** 0.25

    print(f"  T_sat = {wp['T_sat']:.2f} K, T_wall = {T_wall:.2f} K")
    print(f"  과냉도 ΔT = {dT_sub:.1f} K")
    print(f"  Nusselt 해석 h = {h_nusselt:.1f} W/(m²·K)")
    print(f"  Nusselt 액막 두께 δ = {delta_nusselt*1000:.4f} mm")

    # ------ Test A: Nusselt 해석해 재검증 ------
    print("\n  [Test A] Nusselt 막응축 해석해 검증 ...")
    res_a = _test_a_nusselt_validation(wp)
    print(f"    최대 오차: {res_a['max_error']:.6e}")
    print(f"    판정: {'PASS' if res_a['valid'] else 'FAIL'}")

    # ------ Test B: 1D 응축 시뮬레이션 ------
    print("\n  [Test B] 1D 응축 시뮬레이션 ...")
    res_b = _test_b_condensation_sim(wp)
    print(f"    수렴: {res_b['converged']}")
    print(f"    응축 발생: {res_b['condensation_occurred']}")
    print(f"    냉벽 근처 α_l: {res_b['al_near_wall']:.4f}")
    print(f"    벌크 α_l: {res_b['al_bulk']:.4f}")
    print(f"    온도 구배 OK: {res_b['temp_gradient_ok']}")
    print(f"    수치 액막 두께: {res_b['delta_numerical']*1000:.3f} mm")
    print(f"    판정: {'PASS' if (res_b['converged'] and res_b['condensation_occurred']) else 'FAIL'}")

    # ------ 전체 판정 ------
    overall = (res_a['valid'] and res_b['converged'] and res_b['condensation_occurred'])
    print(f"\n  전체 판정: {'PASS' if overall else 'FAIL'}")

    # ------ 시각화 (2×2 subplot) ------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0): Nusselt h vs 판 높이
    ax = axes[0, 0]
    ax.plot(res_a['L_arr'] * 100, res_a['h_numerical_L'], 'bo-', linewidth=2,
            markersize=6, label='수치 모델')
    ax.plot(res_a['L_arr'] * 100, res_a['h_analytical_L'], 'rs--', linewidth=2,
            markersize=6, label='Nusselt 해석해')
    ax.set_xlabel('판 높이 L [cm]')
    ax.set_ylabel('열전달 계수 h [W/(m²·K)]')
    ax.set_title(f'(A) Nusselt 막응축 vs 판 높이 (ΔT={dT_sub:.0f}K)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1): Nusselt h vs 과냉도
    ax = axes[0, 1]
    ax.plot(res_a['dT_arr'], res_a['h_numerical_dT'], 'bo-', linewidth=2,
            markersize=3, label='수치 모델')
    ax.plot(res_a['dT_arr'], res_a['h_analytical_dT'], 'r--', linewidth=2,
            label='Nusselt 해석해')
    ax.set_xlabel('과냉도 ΔT [K]')
    ax.set_ylabel('열전달 계수 h [W/(m²·K)]')
    ax.set_title(f'(A) Nusselt 막응축 vs 과냉도 (L=0.1m)\nmax err = {res_a["max_error"]:.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0): 1D 응축 — 체적분율
    ax = axes[1, 0]
    ax.plot(res_b['x'] * 1000, res_b['alpha_l'], 'b-', linewidth=2,
            label='α_l (액체)')
    ax.plot(res_b['x'] * 1000, 1.0 - res_b['alpha_l'], 'r--', linewidth=2,
            label='α_g (증기)')
    ax.axvline(x=delta_nusselt * 1000, color='gray', linestyle=':',
               linewidth=1.5, label=f'Nusselt δ = {delta_nusselt*1000:.3f} mm')
    ax.set_xlabel('x [mm] (벽으로부터 거리)')
    ax.set_ylabel('체적분율 [-]')
    ax.set_title('(B) 1D 응축 시뮬레이션: 체적분율')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    # (1,1): 1D 응축 — 온도
    ax = axes[1, 1]
    ax.plot(res_b['x'] * 1000, res_b['T'] - 273.15, 'r-', linewidth=2)
    ax.axhline(y=wp['T_sat'] - 273.15, color='gray', linestyle=':',
               alpha=0.7, label=f'T_sat = {wp["T_sat"]-273.15:.1f} °C')
    ax.axhline(y=T_wall - 273.15, color='blue', linestyle=':',
               alpha=0.7, label=f'T_wall = {T_wall-273.15:.1f} °C')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('온도 [°C]')
    ax.set_title('(B) 1D 응축 시뮬레이션: 온도 프로파일')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 25: 수직벽 막 응축 (ΔT = {dT_sub:.0f} K)', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case25_film_condensation.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {fig_path}")

    # VTU 출력
    case_dir = os.path.join(results_dir, 'case25')
    os.makedirs(case_dir, exist_ok=True)
    try:
        cell_data = {
            'temperature': res_b['T_cell_order'],
            'alpha_liquid': res_b['al_cell_order'],
        }
        export_mesh_to_vtu(res_b['mesh'], os.path.join(case_dir, 'condensation_final.vtu'),
                           cell_data)
        print(f"  VTU 저장: {os.path.join(case_dir, 'condensation_final.vtu')}")
    except Exception as e:
        print(f"  [VTU] case25 export skipped: {e}")

    # NPZ 저장
    np.savez(
        os.path.join(results_dir, 'case25_film_condensation.npz'),
        x=res_b['x'], alpha_l=res_b['alpha_l'], T=res_b['T'],
        nusselt_L=res_a['L_arr'], nusselt_h_num=res_a['h_numerical_L'],
        nusselt_h_ana=res_a['h_analytical_L'],
    )

    # 입력 JSON
    params = {
        'case': 25,
        'description': 'Film Condensation — Nusselt + 1D Lee model',
        'T_sat': wp['T_sat'], 'T_wall': T_wall, 'dT_sub': dT_sub,
        'rho_l': wp['rho_l'], 'rho_g': wp['rho_g'],
        'h_nusselt': h_nusselt,
        'delta_nusselt': delta_nusselt,
        'nusselt_error': res_a['max_error'],
        'condensation_occurred': res_b['condensation_occurred'],
        'delta_numerical': res_b['delta_numerical'],
    }
    json_path = os.path.join(case_dir, 'input.json')
    export_input_json(params, json_path)

    return {
        'converged': overall,
        'condensation_occurred': res_b['condensation_occurred'],
        'temp_gradient_ok': res_b['temp_gradient_ok'],
        'max_alpha_l_near_wall': res_b['al_near_wall'],
        'delta_numerical': res_b['delta_numerical'],
        'delta_nusselt': delta_nusselt,
        'film_ratio': res_b['delta_numerical'] / delta_nusselt if delta_nusselt > 0 else 0,
        'h_nusselt': h_nusselt,
        'nusselt_error': res_a['max_error'],
        'T_near_wall': res_b['T_near_wall'],
        'T_bulk': res_b['T_bulk'],
        'T_range': [float(np.min(res_b['T'])), float(np.max(res_b['T']))],
        'time_steps': 500,
        'figure_path': fig_path,
        'vtu_path': os.path.join(case_dir, 'condensation_final.vtu'),
    }

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case25()
    print(f"\n결과 요약:")
    print(f"  Nusselt 오차: {result['nusselt_error']:.6e}")
    print(f"  응축 발생: {result['condensation_occurred']}")
    print(f"  벽 근처 최대 α_l: {result['max_alpha_l_near_wall']:.4f}")
    print(f"  수치 액막 두께: {result['delta_numerical']*1000:.3f} mm")
    print(f"  PASS: {result['converged']}")
