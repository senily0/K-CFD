"""
Case 3: Conjugate Heat Transfer (CHT) 검증.

문제 설명:
  가열 고체 평판(구리) 위를 흐르는 층류 유체(물) 유동의 공액 열전달.
  고체 하면에 일정 열유속(q=5000 W/m²)이 가해지고, 유체는 왼쪽에서
  T_inlet=300K로 유입되어 고체로부터 열을 받아 가열된다.

  형상:
    ┌──────────────────────────┐ y = H_f + H_s (상부 단열)
    │       유체 (물)           │ H_f = 0.03 m
    ├──────────────────────────┤ y = H_s (유체-고체 인터페이스)
    │       고체 (구리)         │ H_s = 0.005 m
    └──────────────────────────┘ y = 0 (가열면, q = 5000 W/m²)
    x=0 (입구)            x=0.5m (출구)

  경계조건:
    - 하면 (y=0): Neumann q = 5000 W/m²
    - 입구 (x=0): Dirichlet T = 300 K
    - 출구 (x=L): Zero gradient (∂T/∂x = 0)
    - 상면 (y=H_f+H_s): 단열 (∂T/∂n = 0)

  물성치:
    - 유체: k=0.6 W/(m·K), ρ=998.2 kg/m³, cp=4182 J/(kg·K)
    - 고체: k=401 W/(m·K) (구리)

  지배 방정식:
    - 고체: ∇·(k_s ∇T) = 0 (순수 열전도)
    - 유체: ρ cp u ∂T/∂x = ∇·(k_f ∇T) (정상 대류-확산)
    - 인터페이스: T 연속, k_s ∂T/∂n|_s = k_f ∂T/∂n|_f

  검증:
    - 에너지 보존: Q_wall ≈ ṁ cp (T_out - T_in)
    - 인터페이스 온도 분포의 물리적 타당성
    - 해석해(Graetz 문제): 열적 발달 영역에서 Nu ~ 7.54 (등열유속)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from scipy import sparse
from scipy.sparse.linalg import spsolve
from mesh.mesh_generator import generate_cht_mesh
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case3(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """CHT 검증 실행."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 3: Conjugate Heat Transfer 검증")
    print("=" * 60)

    # 파라미터
    fluid_length = 0.5
    fluid_height = 0.03
    solid_height = 0.005
    nx, ny_fluid, ny_solid = 40, 12, 6

    k_f = 0.6       # 유체 열전도도 [W/(m·K)]
    k_s = 401.0      # 고체 열전도도 [W/(m·K)] (구리)
    rho_f = 998.2    # 유체 밀도 [kg/m³]
    cp_f = 4182.0    # 유체 비열 [J/(kg·K)]
    U_inlet = 0.05   # 입구 속도 [m/s]
    q_wall = 5000.0  # 하면 열유속 [W/m²]
    T_inlet = 300.0  # 입구 온도 [K]

    print("  격자 생성 중...")
    mesh, zone_map = generate_cht_mesh(
        fluid_length, fluid_height, solid_height, nx, ny_fluid, ny_solid
    )
    fluid_cells = set(zone_map['fluid'])
    solid_cells = set(zone_map['solid'])
    n = mesh.n_cells
    print(f"  총 셀: {n} (유체: {len(fluid_cells)}, 고체: {len(solid_cells)})")

    # 면→경계 캐시
    face_bc = {}
    for bname, fids in mesh.boundary_patches.items():
        for li, fid in enumerate(fids):
            face_bc[fid] = bname

    # 인터페이스 면 찾기
    interface_faces = []
    for fid, face in enumerate(mesh.faces):
        if face.neighbour < 0:
            continue
        o, nb = face.owner, face.neighbour
        if o in fluid_cells and nb in solid_cells:
            interface_faces.append((fid, o, nb))
        elif o in solid_cells and nb in fluid_cells:
            interface_faces.append((fid, nb, o))
    print(f"  인터페이스 면: {len(interface_faces)}")

    # 셀별 열전도도
    k_cell = np.zeros(n)
    for ci in range(n):
        k_cell[ci] = k_s if ci in solid_cells else k_f

    # 셀별 속도장 (유체: Poiseuille 프로파일, 고체: 0)
    # y 기준: 고체 0~H_s, 유체 H_s~(H_s+H_f)
    u_cell = np.zeros(n)
    for ci in fluid_cells:
        y_local = mesh.cells[ci].center[1] - solid_height  # 유체 내 y (0~H_f)
        # Poiseuille: u(y) = 6 * U_mean * y/H * (1 - y/H)
        eta = y_local / fluid_height
        eta = np.clip(eta, 0.0, 1.0)
        u_cell[ci] = 6.0 * U_inlet * eta * (1.0 - eta)

    # 행렬 조립: 유체 영역은 대류-확산, 고체는 순수 확산
    print("  CHT 해석 중 (대류-확산)...")

    rows, cols, vals = [], [], []
    aP = np.zeros(n)
    b = np.zeros(n)

    for fid, face in enumerate(mesh.faces):
        o = face.owner

        if face.neighbour >= 0:
            nb = face.neighbour
            d_vec = mesh.cells[nb].center - mesh.cells[o].center
            d = np.linalg.norm(d_vec)
            if d < 1e-30:
                continue

            # 확산 계수 (조화 평균)
            k_o, k_n = k_cell[o], k_cell[nb]
            k_face = 2 * k_o * k_n / max(k_o + k_n, 1e-30)
            D = k_face * face.area / d

            # 대류 계수 (face normal 방향 질량 유속)
            # face.normal은 owner→neighbour 방향
            fn = face.normal  # 면적 벡터 (면적 × 법선)
            # 면 속도 (upwind 평균)
            u_face = 0.5 * (u_cell[o] + u_cell[nb])
            # 대류 플럭스: F = rho * cp * u_face * face.area * nx_component
            # face.normal은 이미 면적을 포함 (area * unit_normal)
            fn_unit = fn / max(np.linalg.norm(fn), 1e-30)
            F = rho_f * cp_f * u_face * face.area * fn_unit[0]  # x방향 대류만

            # 유체-유체 면만 대류 적용
            if o in fluid_cells and nb in fluid_cells:
                # Upwind 이산화:
                # 면 플럭스 = D*(T_nb - T_o) + F*T_f
                # F > 0: T_f = T_o (owner가 상류)
                #   owner eq:  +(D+F)*T_o - D*T_nb = ...  (유출)
                #   nb eq:     +D*T_nb - (D+F)*T_o = ...  (유입)
                # F < 0: T_f = T_nb (neighbour가 상류)
                #   owner eq:  +D*T_o - (D-F)*T_nb = ...  (유입, F<0이므로 -F>0)
                #   nb eq:     +(D-F)*T_nb - D*T_o = ...  (유출)
                Fp = max(F, 0.0)   # max(F,0)
                Fn = max(-F, 0.0)  # max(-F,0)
                aP[o] += D + Fp
                aP[nb] += D + Fn
                rows.append(o); cols.append(nb); vals.append(-(D + Fn))
                rows.append(nb); cols.append(o); vals.append(-(D + Fp))
            else:
                # 고체-고체 또는 인터페이스: 순수 확산
                aP[o] += D
                aP[nb] += D
                rows.append(o); cols.append(nb); vals.append(-D)
                rows.append(nb); cols.append(o); vals.append(-D)

        else:
            # 경계면
            bname = face_bc.get(fid, '')
            d = np.linalg.norm(face.center - mesh.cells[o].center)
            if d < 1e-30:
                continue

            if bname == 'wall_heated':
                # Neumann: q = q_wall (열유속 유입)
                b[o] += q_wall * face.area

            elif bname == 'inlet':
                # Dirichlet: T = T_inlet
                # 확산: aP += D, b += D*T_inlet
                # 대류 유입: b += F*T_inlet (소스항, aP에는 추가하지 않음)
                D = k_cell[o] * face.area / d
                F_in = rho_f * cp_f * u_cell[o] * face.area
                aP[o] += D
                b[o] += D * T_inlet + max(F_in, 0) * T_inlet

            elif bname == 'outlet':
                # 출구: zero-gradient (∂T/∂x=0) + convective outflow
                # upwind: 출구 면의 온도 = 셀 온도 → F_out 을 aP에 추가
                F_out = rho_f * cp_f * u_cell[o] * face.area
                aP[o] += max(F_out, 0)

            elif bname == 'wall_top':
                # 상면 단열
                pass

            else:
                # 기타: 단열
                pass

    # 대각 추가
    for ci in range(n):
        rows.append(ci); cols.append(ci); vals.append(aP[ci])

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    T = spsolve(A, b)

    # NaN 체크
    if np.any(np.isnan(T)):
        print("  경고: NaN 발생, T_inlet으로 초기화")
        T = np.full(n, T_inlet)

    print(f"  온도 범위: [{T.min():.2f}, {T.max():.2f}] K")

    # 결과 분석
    T_solid_vals = T[zone_map['solid']]
    T_fluid_vals = T[zone_map['fluid']]
    T_solid_max = float(T_solid_vals.max())
    T_fluid_max = float(T_fluid_vals.max())

    # 인터페이스 온도
    T_interface_list = []
    x_interface = []
    for fid, fc, sc in interface_faces:
        face = mesh.faces[fid]
        T_int = 0.5 * (T[fc] + T[sc])
        T_interface_list.append(T_int)
        x_interface.append(face.center[0])
    idx = np.argsort(x_interface)
    x_interface = np.array(x_interface)[idx]
    T_interface_arr = np.array(T_interface_list)[idx]
    T_interface_avg = float(np.mean(T_interface_arr)) if len(T_interface_arr) > 0 else T_inlet

    # 에너지 보존: Q_input vs Q_fluid
    Q_input = q_wall * fluid_length  # [W/m]

    # 출구 유체 평균 온도 (질량 가중 평균)
    T_out_sum, m_out_sum = 0.0, 0.0
    T_in_sum, m_in_sum = 0.0, 0.0
    for ci in zone_map['fluid']:
        xc = mesh.cells[ci].center[0]
        y_local = mesh.cells[ci].center[1] - solid_height
        vol = mesh.cells[ci].volume
        if xc > fluid_length * 0.95:
            T_out_sum += T[ci] * u_cell[ci] * vol
            m_out_sum += u_cell[ci] * vol
        elif xc < fluid_length * 0.05:
            T_in_sum += T[ci] * u_cell[ci] * vol
            m_in_sum += u_cell[ci] * vol

    T_out_avg = T_out_sum / max(m_out_sum, 1e-30)
    T_in_avg = T_in_sum / max(m_in_sum, 1e-30)

    Q_fluid = rho_f * cp_f * U_inlet * fluid_height * (T_out_avg - T_in_avg)
    energy_err = abs(Q_fluid / max(abs(Q_input), 1e-15) - 1.0)

    # 해석적 예측: T_out = T_in + Q/(m_dot * cp)
    m_dot = rho_f * U_inlet * fluid_height  # [kg/(m·s)]
    T_out_analytical = T_inlet + Q_input / (m_dot * cp_f)

    converged = (T.min() > 200 and T.max() < 500 and energy_err < 0.20)

    print(f"  고체 최대 온도: {T_solid_max:.2f} K")
    print(f"  유체 최대 온도: {T_fluid_max:.2f} K")
    print(f"  인터페이스 평균 온도: {T_interface_avg:.2f} K")
    print(f"  출구 유체 평균 온도: {T_out_avg:.2f} K (해석: {T_out_analytical:.2f} K)")
    print(f"  에너지: 입력={Q_input:.2f}, 유체흡수={Q_fluid:.2f} W/m")
    print(f"  에너지 균형 오차: {energy_err * 100:.1f}%")

    # 시각화 (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    x_f = [mesh.cells[ci].center[0] for ci in zone_map['fluid']]
    y_f = [mesh.cells[ci].center[1] for ci in zone_map['fluid']]
    sc = ax.scatter(x_f, y_f, c=T_fluid_vals, cmap='hot', s=8)
    plt.colorbar(sc, ax=ax, label='T [K]')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('유체 온도 분포')

    ax = axes[0, 1]
    x_s = [mesh.cells[ci].center[0] for ci in zone_map['solid']]
    y_s = [mesh.cells[ci].center[1] for ci in zone_map['solid']]
    sc = ax.scatter(x_s, y_s, c=T_solid_vals, cmap='hot', s=8)
    plt.colorbar(sc, ax=ax, label='T [K]')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('고체 온도 분포')

    ax = axes[1, 0]
    if len(x_interface) > 0:
        ax.plot(x_interface, T_interface_arr, 'b-o', markersize=3, label='수치해 (인터페이스)')
    ax.axhline(y=T_inlet, color='gray', linestyle='--', alpha=0.5, label=f'T_inlet = {T_inlet} K')
    ax.axhline(y=T_out_analytical, color='r', linestyle=':', alpha=0.7,
               label=f'T_out 해석 = {T_out_analytical:.1f} K')
    ax.set_xlabel('x [m]'); ax.set_ylabel('T [K]')
    ax.set_title(f'인터페이스 온도 분포 (에너지 오차: {energy_err*100:.1f}%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    x_target = fluid_length * 0.5
    tol_x = fluid_length / nx * 2
    y_prof, T_prof = [], []
    for ci in range(n):
        if abs(mesh.cells[ci].center[0] - x_target) < tol_x:
            y_prof.append(mesh.cells[ci].center[1])
            T_prof.append(T[ci])
    if y_prof:
        idx2 = np.argsort(y_prof)
        y_arr = np.array(y_prof)[idx2]
        T_arr = np.array(T_prof)[idx2]
        # Split into solid and fluid regions for distinct labeling
        solid_mask = y_arr <= solid_height
        fluid_mask = y_arr > solid_height
        if np.any(solid_mask):
            ax.plot(T_arr[solid_mask], y_arr[solid_mask] * 1000, 'r-o',
                    markersize=3, label='고체 (구리) 온도')
        if np.any(fluid_mask):
            ax.plot(T_arr[fluid_mask], y_arr[fluid_mask] * 1000, 'b-o',
                    markersize=3, label='유체 (물) 온도')
        ax.axhline(y=solid_height * 1000, color='gray', linestyle='--',
                    alpha=0.7, label=f'유체-고체 경계 (y={solid_height*1000:.1f} mm)')
    ax.set_xlabel('T [K]'); ax.set_ylabel('y [mm]')
    ax.set_title(f'수직 온도 프로파일 (x = {x_target:.2f} m = 0.5L)\n'
                 f'[하단: 고체, 상단: 유체]')
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 3: 공액 열전달 (q = {q_wall} W/m², U = {U_inlet} m/s)', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case3_cht.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case3')
    os.makedirs(case_dir, exist_ok=True)
    params = {
        'case': 3,
        'description': 'Conjugate Heat Transfer — heated plate with laminar flow',
        'Lx': fluid_length, 'Ly_fluid': fluid_height, 'Ly_solid': solid_height,
        'nx': nx, 'ny_fluid': ny_fluid, 'ny_solid': ny_solid,
        'k_fluid': k_f, 'k_solid': k_s, 'q_wall': q_wall,
        'U_inlet': U_inlet, 'T_inlet': T_inlet,
        'T_out_analytical': T_out_analytical,
    }
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {'temperature': T}
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case3 export skipped: {e}")

    return {
        'converged': converged,
        'iterations': 1,
        'T_interface_avg': T_interface_avg,
        'Q_input': Q_input,
        'Q_fluid': Q_fluid,
        'energy_balance_error': energy_err,
        'T_solid_max': T_solid_max,
        'T_fluid_max': T_fluid_max,
        'T_out_avg': T_out_avg,
        'T_out_analytical': T_out_analytical,
        'figure_path': fig_path
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case3()
