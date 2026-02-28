"""
Case 9: Stefan 문제 기반 상변화 (Lee 모델) 검증.

1D 도메인 [0, L], 초기 전체 액체 (T = T_sat).
왼쪽 벽: T = T_hot > T_sat -> 증발 전선이 오른쪽으로 이동.

해석 해 (Stefan 문제):
    s(t) = 2 * lambda * sqrt(alpha_th * t)
    lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi)
    Ste = cp * (T_hot - T_sat) / L_latent
    alpha_th = k / (rho * cp)

검증 방법:
    - Lee 모델로 alpha, T 를 시간 전진
    - alpha = 0.5 등위선을 인터페이스로 정의
    - 해석 해와 L2 오차 비교
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
from scipy.special import erf
from scipy.optimize import brentq

from mesh.mesh_generator import _make_structured_quad_mesh
from mesh.mesh_reader import build_fvmesh_from_arrays
from core.fields import ScalarField
from models.phase_change import LeePhaseChangeModel
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json

# ---------------------------------------------------------------------------
# 1D 채널 메쉬 생성 (nx x 1 quad cells)
# ---------------------------------------------------------------------------

def _make_1d_mesh(L: float, nx: int):
    """길이 L, 셀 수 nx 인 1D (x 방향) 채널 메쉬."""
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, L, L / nx,   # height = dx (정사각형 셀)
        nx, 1,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'wall_top',
            'left': 'left', 'right': 'right'
        }
    )
    return build_fvmesh_from_arrays(nodes, cells, bfaces)

# ---------------------------------------------------------------------------
# 해석 해: Stefan 문제 lambda 계산
# ---------------------------------------------------------------------------

def _stefan_lambda(Ste: float) -> float:
    """
    lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi) 를 만족하는 lambda.
    Ste = cp * (T_hot - T_sat) / L_latent (Stefan number).
    """
    rhs = Ste / np.sqrt(np.pi)

    def f(lam):
        if lam < 1e-12:
            return lam - rhs
        return lam * np.exp(lam**2) * erf(lam) - rhs

    hi = 0.1
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            return hi
    lam = brentq(f, 1e-9, hi, xtol=1e-12, maxiter=200)
    return lam

def stefan_interface_position(t: float, lam: float, alpha_th: float) -> float:
    """s(t) = 2 * lambda * sqrt(alpha_th * t)."""
    if t <= 0.0:
        return 0.0
    return 2.0 * lam * np.sqrt(alpha_th * t)

# ---------------------------------------------------------------------------
# 인터페이스 위치 추출 (alpha = 0.5 등위선 + 초기 단계 보조 추정)
# ---------------------------------------------------------------------------

def _find_interface(x: np.ndarray, alpha_l: np.ndarray,
                    dx: float = None) -> float:
    """
    alpha_l 프로파일로부터 기체-액체 인터페이스 x 위치 추정.

    주 방법: alpha_l = 0.5 등위선 선형 보간 (정확한 계면 위치).
    보조 방법 (초기 단계, 아직 alpha_l = 0.5 미도달):
        가장 많이 증발한 셀의 alpha_gas 를 이용한 외삽.
        x_intf ≈ x[i_min] * (1 - alpha_l[i_min])  (왼쪽 끝 기준 선형 외삽)

    케이스 방향: 왼쪽 벽이 고온 -> 왼쪽(기체, alpha 낮음), 오른쪽(액체, alpha 높음).
    """
    threshold = 0.5
    n = len(alpha_l)

    # 방법 1: alpha_l = 0.5 등위선 선형 보간
    if not np.all(alpha_l >= threshold):
        for i in range(n - 1):
            a0, a1 = alpha_l[i], alpha_l[i + 1]
            if a0 <= threshold < a1:
                if abs(a1 - a0) < 1e-12:
                    return float(x[i + 1])
                frac = (threshold - a0) / (a1 - a0)
                return float(x[i] + frac * (x[i + 1] - x[i]))
        # 모든 셀이 기체 — 전선이 도메인 끝을 넘어섬
        return float(x[-1])

    # 방법 2 (초기 단계): 아직 어떤 셀도 alpha_l = 0.5 미만으로 내려가지 않은 경우.
    # 가장 많이 증발한 셀(alpha_l 최솟값)을 찾아 alpha_gas 비율로 위치 추정.
    # x_intf ≈ alpha_gas_max / (alpha_gas_max + alpha_l_right_neighbor) * dx + x[i_min]
    # 단순화: 왼쪽 끝 셀부터 증발이 시작되므로, alpha_gas 합계로 추정
    alpha_gas = 1.0 - alpha_l
    dx_val = dx if dx is not None else (x[1] - x[0] if n > 1 else 1.0)
    total_gas_vol = float(np.sum(alpha_gas)) * dx_val
    return total_gas_vol

# ---------------------------------------------------------------------------
# 메인 케이스 함수
# ---------------------------------------------------------------------------

def run_case9(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """
    Stefan 문제 상변화 검증 실행.

    Returns
    -------
    result : dict with keys
        'L2_error'    : float - 인터페이스 위치 L2 오차 (상대, 무차원)
        'converged'   : bool
        'figure_path' : str
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 9: Stefan 문제 (Lee 모델) 상변화 검증")
    print("=" * 60)

    # --- 물성치 및 파라미터 ---
    L        = 0.01       # 도메인 길이 [m]
    nx       = 100        # 셀 수
    T_sat    = 373.15     # 포화 온도 [K]
    T_hot    = 383.15     # 가열 벽 온도 [K]  (dT = 10 K)
    rho      = 1000.0     # 액체 밀도 [kg/m3]
    cp       = 4200.0     # 비열 [J/(kg*K)]
    k        = 0.6        # 열전도도 [W/(m*K)]
    L_lat    = 2260000.0  # 잠열 [J/kg]
    rho_g    = 0.6        # 증기 밀도 [kg/m3]

    # Lee 계수: 물리적 인터페이스 이동 속도(dx/v_intf ~ 134 스텝)와 일치하도록 설정.
    # r_evap = 10 -> d_alpha/step ~0.0075 -> 셀 하나가 ~133 스텝에 걸쳐 증발
    # -> alpha 프로파일이 여러 셀에 걸쳐 분포 -> threshold 추적이 연속적으로 작동.
    r_evap   = 10.0

    alpha_th = k / (rho * cp)
    Ste      = cp * (T_hot - T_sat) / L_lat
    dx       = L / nx

    print(f"  Stefan number Ste     = {Ste:.6f}")
    print(f"  열확산계수 alpha_th   = {alpha_th:.3e} m2/s")

    # --- 해석 해 계수 ---
    lam_stefan = _stefan_lambda(Ste)
    print(f"  Stefan lambda         = {lam_stefan:.6f}")

    # --- 메쉬 ---
    mesh = _make_1d_mesh(L, nx)
    n    = mesh.n_cells

    # 셀 중심 x 좌표 및 정렬 인덱스
    x_cells  = np.array([mesh.cells[ci].center[0] for ci in range(n)])
    sort_idx = np.argsort(x_cells)
    x_sorted = x_cells[sort_idx]

    print(f"  메쉬: {n} cells, dx = {dx:.4e} m")

    # --- Lee 모델 인스턴스 ---
    pc = LeePhaseChangeModel(
        mesh, T_sat=T_sat,
        r_evap=r_evap, r_cond=r_evap,
        L_latent=L_lat,
        rho_l=rho, rho_g=rho_g
    )

    # --- 필드 초기화 ---
    T_field = ScalarField(mesh, "T",       default=T_sat)
    alpha_l = ScalarField(mesh, "alpha_l", default=1.0)

    T_field.set_boundary('left',  T_hot)
    T_field.set_boundary('right', T_sat)

    # --- 시간 간격 ---
    # 열확산 안정성: dt < 0.5 * dx^2 / alpha_th
    dt = 0.4 * dx**2 / alpha_th

    # 인터페이스가 도메인 약 40% 까지 이동하는 시간
    t_end_est = (0.4 * L / (2.0 * lam_stefan))**2 / alpha_th
    t_end = min(t_end_est, 300.0 * dt)   # 최대 300 스텝 분량
    t_end = max(t_end, 30.0 * dt)        # 최소 30 스텝
    n_steps = max(int(t_end / dt), 30)
    dt = t_end / n_steps

    print(f"  dt = {dt:.4e} s, n_steps = {n_steps}, t_end = {t_end:.4e} s")

    # --- 내부면/경계면 정보 사전 계산 ---
    internal_face_data = []   # (owner, nb, area, d, vol_o, vol_nb)
    left_bc_data       = []   # (owner, d, area)
    right_bc_data      = []   # (owner, d, area)
    cell_volumes       = np.array([mesh.cells[ci].volume for ci in range(n)])

    for fid in range(mesh.n_faces):
        face  = mesh.faces[fid]
        owner = face.owner
        nb    = face.neighbour
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
            # wall_bottom, wall_top: adiabatic

    # --- 기록 배열 ---
    t_history    = []
    s_numerical  = []
    s_analytical = []

    # --- 시간 전진 루프 ---
    t = 0.0
    for step in range(n_steps):

        T_vals  = T_field.values
        al_vals = alpha_l.values

        # 유효 rho*cp 배열
        rho_cp = (al_vals * rho + (1.0 - al_vals) * rho_g) * cp

        # 1) 열확산 (명시적 오일러)
        dT = np.zeros(n)

        for (o, nb, area, d, vol_o, vol_nb) in internal_face_data:
            flux    = k * area * (T_vals[nb] - T_vals[o]) / d
            dT[o]  += dt * flux / (rho_cp[o]  * vol_o)
            dT[nb] -= dt * flux / (rho_cp[nb] * vol_nb)

        for (o, d, area) in left_bc_data:
            flux   = k * area * (T_hot - T_vals[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        for (o, d, area) in right_bc_data:
            flux   = k * area * (T_sat - T_vals[o]) / d
            dT[o] += dt * flux / (rho_cp[o] * cell_volumes[o])

        T_field.values = T_vals + dT

        # 2) Lee 모델 소스항
        src = pc.get_source_terms(T_field, alpha_l)

        # alpha 업데이트 (명시적 오일러)
        alpha_l.values += dt * src['alpha_l']
        alpha_l.values  = np.clip(alpha_l.values, 0.0, 1.0)

        # 잠열 에너지 소스 -> 온도 변화
        # rho_cp 최솟값을 rho_g*cp 로 제한하여 수치 발산 방지
        rho_cp_new = (alpha_l.values * rho + (1.0 - alpha_l.values) * rho_g) * cp
        rho_cp_min = rho_g * cp  # 기체 단독 시 최솟값
        dT_lat = dt * src['energy'] / np.maximum(rho_cp_new, rho_cp_min)
        # 잠열에 의한 온도 변화량을 ±(T_hot - T_sat) 로 제한 (수치 불안정 방지)
        dT_max = T_hot - T_sat
        dT_lat = np.clip(dT_lat, -dT_max, dT_max)
        T_field.values = T_field.values + dT_lat
        # T_sat 아래로 내려가지 않도록 (단방향 증발 케이스)
        T_field.values = np.maximum(T_field.values, T_sat - 0.1)

        # 3) 왼쪽 Dirichlet 경계 재적용
        for (o, d, area) in left_bc_data:
            T_field.values[o] = T_hot

        t += dt

        # 4) 인터페이스 위치 기록 (alpha_gas 체적 가중 평균)
        al_sorted = alpha_l.values[sort_idx]
        s_num     = _find_interface(x_sorted, al_sorted, dx=dx)
        s_ana     = stefan_interface_position(t, lam_stefan, alpha_th)

        t_history.append(t)
        s_numerical.append(s_num)
        s_analytical.append(s_ana)

        if step % max(1, n_steps // 5) == 0:
            print(f"  step {step:4d}/{n_steps}, t={t:.4e}s, "
                  f"s_num={s_num*1e3:.3f}mm, s_ana={s_ana*1e3:.3f}mm")

    t_arr     = np.array(t_history)
    s_num_arr = np.array(s_numerical)
    s_ana_arr = np.array(s_analytical)

    # --- L2 오차 계산 ---
    # 초기 격자 해상도 이하 구간은 무의미하므로 제외 (3*dx 이상부터 유효)
    valid = (s_ana_arr > 3 * dx) & (s_ana_arr < 0.8 * L) & (s_num_arr > 0)
    if np.sum(valid) > 2:
        rel_err  = np.abs(s_num_arr[valid] - s_ana_arr[valid]) / np.maximum(s_ana_arr[valid], 1e-12)
        L2_error = float(np.sqrt(np.mean(rel_err**2)))
    else:
        # 인터페이스가 아직 해석 가능 범위에 도달하지 못함 — 최종값으로 계산
        if s_ana_arr[-1] > dx and s_num_arr[-1] > 0:
            L2_error = float(abs(s_num_arr[-1] - s_ana_arr[-1]) / s_ana_arr[-1])
        else:
            L2_error = 1.0

    # 인터페이스가 최소 2*dx 이상 이동했는지 확인 (의미 있는 이동)
    moved_right = bool(len(s_num_arr) > 1 and s_num_arr[-1] > 2.0 * dx)
    # 최종 위치가 해석해 대비 20% 이내인지 확인
    s_final_num = float(s_num_arr[-1])
    s_final_ana = float(s_ana_arr[-1])
    final_rel_err = (abs(s_final_num - s_final_ana) / max(s_final_ana, 1e-12)
                     if s_final_ana > dx else 1.0)
    within_20pct = bool(final_rel_err < 0.20)
    # PASS 기준: L2 < 0.10 (해석해 대비 10% 이내) AND 최종 위치 20% 이내
    converged   = moved_right and (L2_error < 0.10) and within_20pct

    print(f"\n  최종 수치 인터페이스 위치: {s_final_num*1e3:.4f} mm")
    print(f"  최종 해석 인터페이스 위치: {s_final_ana*1e3:.4f} mm")
    print(f"  최종 상대 오차:           {final_rel_err:.4f}")
    print(f"  L2 오차 (상대):           {L2_error:.4f}")
    print(f"  최종 20% 이내:            {within_20pct}")
    print(f"  수렴 여부 (PASS):         {converged}")

    # --- 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 인터페이스 위치 vs 시간
    ax = axes[0]
    ax.plot(t_arr * 1e3, s_num_arr * 1e3, 'b-',  linewidth=2,
            label='Lee 모델 (alpha_gas 체적 가중 평균)')
    ax.plot(t_arr * 1e3, s_ana_arr * 1e3, 'r--', linewidth=2, label='Stefan 해석 해')
    ax.set_xlabel('시간 [ms]')
    ax.set_ylabel('인터페이스 위치 [mm]')
    ax.set_title('Stefan 문제: 인터페이스 위치 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)
    pass_str = 'PASS' if converged else 'FAIL'
    ax.text(0.05, 0.80,
            f'L2 오차 = {L2_error:.3f}\n최종 오차 = {final_rel_err:.3f}\n{pass_str}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 최종 온도 및 체적분율 프로파일
    ax  = axes[1]
    T_sorted  = T_field.values[sort_idx]
    al_sorted = alpha_l.values[sort_idx]
    ax2 = ax.twinx()
    lns1 = ax.plot(x_sorted * 1e3, T_sorted - 273.15, 'r-',
                   linewidth=2, label='T [degC]')
    lns2 = ax2.plot(x_sorted * 1e3, al_sorted, 'b--',
                    linewidth=2, label='alpha_l [-]')
    ax.axhline(y=T_sat - 273.15, color='gray', linestyle=':',
               alpha=0.7, label=f'T_sat = {T_sat - 273.15:.1f} degC')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('온도 [degC]', color='r')
    ax2.set_ylabel('액체 체적분율 alpha_l', color='b')
    ax2.set_ylim(-0.05, 1.15)
    ax.set_title(f'최종 프로파일 (t = {t_arr[-1]*1e3:.2f} ms)')
    lns  = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case9_phase_change.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # 결과 저장
    np.savez(
        os.path.join(results_dir, 'case9_phase_change.npz'),
        t=t_arr,
        s_numerical=s_num_arr,
        s_analytical=s_ana_arr,
        x=x_sorted,
        T_final=T_sorted,
        alpha_l_final=al_sorted,
    )

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case9')
    os.makedirs(case_dir, exist_ok=True)
    params = {'L': L, 'nx': nx, 'T_sat': T_sat, 'T_hot': T_hot,
              'L_latent': L_lat, 'k': k, 'rho': rho, 'cp': cp}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'temperature': T_field.values,
            'alpha_l': alpha_l.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case9 export skipped: {e}")

    return {
        'L2_error':           L2_error,
        'converged':          converged,
        'figure_path':        fig_path,
        's_final_numerical':  s_final_num,
        's_final_analytical': s_final_ana,
        'final_rel_error':    final_rel_err,
        'within_20pct':       within_20pct,
        'stefan_lambda':      float(lam_stefan),
        'Ste':                float(Ste),
    }

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case9()
    print(f"\n결과 요약:")
    print(f"  Stefan number:           {result['Ste']:.6f}")
    print(f"  Stefan lambda:           {result['stefan_lambda']:.6f}")
    print(f"  최종 수치 인터페이스:    {result['s_final_numerical']*1e3:.4f} mm")
    print(f"  최종 해석 인터페이스:    {result['s_final_analytical']*1e3:.4f} mm")
    print(f"  최종 상대 오차:          {result['final_rel_error']:.4f}")
    print(f"  20% 이내:                {result['within_20pct']}")
    print(f"  L2 오차:                 {result['L2_error']:.4f}")
    print(f"  수렴 (PASS):             {result['converged']}")
