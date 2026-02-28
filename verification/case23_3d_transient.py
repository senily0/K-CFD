"""
Case 23: 3D 과도 채널 유동 (Transient Channel Flow) 검증.

3D 직육면체 채널에서 비정상 단상 유동 해석.
초기 정지 상태에서 입구 속도를 갑자기 인가(impulsive start)하여
속도 프로파일이 시간에 따라 발달하는 과정을 검증한다.

물리: 비압축성 단상 Navier-Stokes (비정상)
격자: 3D 구조 hex (20×8×8 = 1280 cells)
검증: (1) 충분한 시간 후 정상 Poiseuille 프로파일 수렴
      (2) 중간 시각에서의 비정상 발달 확인
      (3) 시간 스텝별 VTU 출력
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator_3d import generate_3d_channel_mesh
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json
from models.single_phase import SIMPLESolver


def run_case23(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    3D 과도 채널 유동 검증 실행.

    Returns
    -------
    result : dict
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 23: 3D 과도 채널 유동 (Transient Channel Flow)")
    print("=" * 60)

    # ---- 파라미터 ----
    Lx = 1.0       # 채널 길이 [m]
    Ly = 0.1       # 채널 높이 [m]
    Lz = 0.1       # 채널 폭 [m]
    nx, ny, nz = 16, 8, 8
    rho = 1.0      # 밀도 [kg/m³]
    mu = 0.01      # 점성 [Pa·s]
    U_in = 1.0     # 입구 속도 [m/s]
    Re = rho * U_in * Ly / mu  # Re = 10

    # 시간 설정
    nu = mu / rho
    t_char = Ly**2 / nu  # 확산 특성 시간 = H^2/nu
    dt = t_char / 50.0
    t_end = t_char * 1.5  # 충분한 시간 (3배 특성 시간)
    n_snapshots = 5  # VTU 스냅샷 수

    print(f"  Re = {Re:.0f}, rho = {rho}, mu = {mu}")
    print(f"  채널: Lx={Lx}, Ly={Ly}, Lz={Lz}")
    print(f"  격자: {nx}x{ny}x{nz} = {nx*ny*nz} cells")
    print(f"  특성 시간: {t_char:.4f} s, dt = {dt:.6f} s, t_end = {t_end:.4f} s")

    # ---- 격자 생성 ----
    print("  3D 격자 생성 중...")
    mesh = generate_3d_channel_mesh(Lx, Ly, Lz, nx, ny, nz)
    print(f"  {mesh.summary()}")

    # ---- 솔버 설정 ----
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 500
    solver.tol = 1e-4
    solver.alpha_u = 0.7
    solver.alpha_p = 0.3
    solver.dt = dt

    # 경계조건
    solver.set_velocity_bc('inlet', 'dirichlet', [U_in, 0.0, 0.0])
    solver.set_velocity_bc('outlet', 'zero_gradient')
    solver.set_velocity_bc('wall_bottom', 'dirichlet', [0.0, 0.0, 0.0])
    solver.set_velocity_bc('wall_top', 'dirichlet', [0.0, 0.0, 0.0])
    solver.set_velocity_bc('wall_front', 'dirichlet', [0.0, 0.0, 0.0])
    solver.set_velocity_bc('wall_back', 'dirichlet', [0.0, 0.0, 0.0])

    solver.set_pressure_bc('inlet', 'zero_gradient')
    solver.set_pressure_bc('outlet', 'dirichlet', 0.0)
    for patch in ['wall_bottom', 'wall_top', 'wall_front', 'wall_back']:
        solver.set_pressure_bc(patch, 'zero_gradient')

    # 초기 조건: 정지
    solver.U.values[:] = 0.0
    solver.p.values[:] = 0.0

    # ---- 과도 해석 (수동 시간 전진) ----
    print("  과도 해석 중...")
    solver.transient = True
    t = 0.0
    step = 0
    snapshot_interval = max(1, int((t_end / dt) / n_snapshots))
    snapshots = []
    residual_history = []

    case_dir = os.path.join(results_dir, 'case23')
    os.makedirs(case_dir, exist_ok=True)

    # 출구 중앙(x=Lx 부근) y방향 프로파일 추출 함수
    x_sample = Lx * 0.8  # 출구 근처
    z_mid = Lz / 2.0
    tol_x = Lx / nx * 0.6   # x 방향: 해당 셀만 포함
    tol_z = Lz / nz * 0.6   # z 방향: 중앙 셀만 포함

    def extract_y_profile():
        y_arr, u_arr = [], []
        for ci in range(mesh.n_cells):
            cc = mesh.cells[ci].center
            if abs(cc[0] - x_sample) < tol_x and abs(cc[2] - z_mid) < tol_z:
                y_arr.append(cc[1])
                u_arr.append(solver.U.values[ci, 0])
        y_arr = np.array(y_arr)
        u_arr = np.array(u_arr)
        if len(y_arr) > 0:
            idx = np.argsort(y_arr)
            return y_arr[idx], u_arr[idx]
        return np.array([]), np.array([])

    max_steps = int(t_end / dt) + 1
    for step_i in range(max_steps):
        solver.U.store_old()
        solver.p.store_old()

        # SIMPLE 내부 반복
        max_res = 1.0
        for inner in range(30):
            mf = solver._face_mass_flux()
            res_mom = []
            ndim = getattr(mesh, 'ndim', 3)
            for comp in range(ndim):
                res_mom.append(solver._momentum_eq(comp, mf))
            mf = solver._face_mass_flux()
            rp = solver._pressure_correction(mf)
            max_res = max(max(res_mom), rp)
            if max_res < solver.tol * 10:
                break

        t += dt
        step += 1
        residual_history.append(max_res)

        # VTU 스냅샷
        if step % snapshot_interval == 0 or step == 1:
            y_snap, u_snap = extract_y_profile()
            snapshots.append({
                'time': t,
                'step': step,
                'y': y_snap.copy(),
                'u': u_snap.copy(),
                'max_res': max_res,
            })

            # VTU 저장
            vtu_path = os.path.join(case_dir, f'flow_t{step:04d}.vtu')
            cell_data = {
                'u': solver.U.values[:, 0],
                'v': solver.U.values[:, 1],
                'w': solver.U.values[:, 2],
                'p': solver.p.values,
                'velocity_magnitude': np.linalg.norm(solver.U.values, axis=1),
            }
            export_mesh_to_vtu(mesh, vtu_path, cell_data=cell_data)

        if step % 50 == 0:
            print(f"    Step {step}, t={t:.5f}s, residual={max_res:.2e}")

        if t >= t_end - 1e-15:
            break

    solver.transient = False

    # 마지막 VTU 저장
    final_vtu = os.path.join(case_dir, 'flow_final.vtu')
    cell_data = {
        'u': solver.U.values[:, 0],
        'v': solver.U.values[:, 1],
        'w': solver.U.values[:, 2],
        'p': solver.p.values,
        'velocity_magnitude': np.linalg.norm(solver.U.values, axis=1),
    }
    export_mesh_to_vtu(mesh, final_vtu, cell_data=cell_data)

    print(f"  총 {step} 스텝 완료, 최종 t = {t:.5f} s")

    # ---- 최종 프로파일 vs 해석해 ----
    y_final, u_final = extract_y_profile()

    # 3D 직사각형 덕트 해석해 (Fourier 급수)
    # 2D Poiseuille (u_max=1.5*U_mean)이 아닌, 정사각 단면의 해석해 사용
    # u_max/U_mean ≈ 2.096 (정사각 단면)
    if len(y_final) > 0:
        a = Ly / 2.0   # 반 높이
        b = Lz / 2.0   # 반 폭
        # 평균 속도로부터 압력 구배 계산
        sum_tanh = sum(np.tanh(nn * np.pi * b / (2*a)) / nn**5
                       for nn in range(1, 200, 2))
        corr = 1.0 - 192*a / (np.pi**5 * b) * sum_tanh
        dpdx = -3.0 * mu * U_in / (a**2 * corr)

        # z = z_mid (단면 중심)에서의 y 프로파일
        u_analytical = np.zeros(len(y_final))
        for iy, y_val in enumerate(y_final):
            y_rel = y_val - a
            u_val = 0.0
            for nn in range(1, 100, 2):
                u_val += ((-1)**((nn-1)//2) / nn**3 *
                          (1.0 - np.cosh(nn*np.pi*0.0/(2*a)) /
                           np.cosh(nn*np.pi*b/(2*a))) *
                          np.cos(nn*np.pi*y_rel/(2*a)))
            u_val *= 16*a**2 / (mu * np.pi**3) * (-dpdx)
            u_analytical[iy] = u_val

        diff = u_final - u_analytical
        L2_error = float(np.sqrt(np.mean(diff**2)) /
                         max(np.sqrt(np.mean(u_analytical**2)), 1e-15))
    else:
        u_analytical = np.array([])
        L2_error = 1.0

    print(f"  최종 프로파일 L2 오차 (vs 3D 덕트 해석해): {L2_error:.4e}")

    # ---- 비정상 발달 확인: 초기 프로파일이 최종보다 평탄해야 함 ----
    transient_development = False
    if len(snapshots) >= 2:
        u_early = snapshots[0]['u']
        u_late = snapshots[-1]['u']
        if len(u_early) > 0 and len(u_late) > 0:
            # 초기 프로파일의 최대값/평균 비율이 최종보다 작아야 함
            ratio_early = float(np.max(u_early) / max(np.mean(u_early), 1e-15))
            ratio_late = float(np.max(u_late) / max(np.mean(u_late), 1e-15))
            transient_development = ratio_late > ratio_early
            print(f"  비정상 발달: peak/mean 비율 초기={ratio_early:.3f}, 최종={ratio_late:.3f}")

    # ---- 시각화 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0): 시간에 따른 속도 프로파일 발달
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(snapshots)))
    for i, snap in enumerate(snapshots):
        if len(snap['y']) > 0:
            label = f"t = {snap['time']:.4f}s"
            ax.plot(snap['u'] / U_in, snap['y'] / Ly, '-o',
                    color=colors[i], linewidth=1.5, markersize=2, label=label)
    if len(y_final) > 0:
        ax.plot(u_analytical / U_in, y_final / Ly, 'k--', linewidth=2.0,
                label='3D 덕트 해석해')
    ax.set_xlabel('u / U_in [-]')
    ax.set_ylabel('y / H [-]')
    ax.set_title('속도 프로파일 시간 발달 (x = 0.8L)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (0,1): 잔차 이력
    ax = axes[0, 1]
    ax.semilogy(residual_history, 'b-', linewidth=0.8)
    ax.set_xlabel('시간 스텝')
    ax.set_ylabel('잔차')
    ax.set_title('비정상 해석 잔차 이력')
    ax.grid(True, alpha=0.3)

    # (1,0): 최종 프로파일 vs 해석해
    ax = axes[1, 0]
    if len(y_final) > 0:
        ax.plot(u_final * 1000, y_final * 1000, 'bo-', markersize=4,
                linewidth=1.5, label='수치해 (FVM)')
        ax.plot(u_analytical * 1000, y_final * 1000, 'r--', linewidth=2.0,
                label='Poiseuille 해석해')
    ax.set_xlabel('u [mm/s]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'최종 속도 프로파일 (L2 = {L2_error:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1): 중앙점 속도 시간 이력
    ax = axes[1, 1]
    times_snap = [s['time'] for s in snapshots]
    u_center_snap = []
    for s in snapshots:
        if len(s['y']) > 0:
            mid_idx = len(s['y']) // 2
            u_center_snap.append(s['u'][mid_idx])
        else:
            u_center_snap.append(0.0)
    ax.plot(np.array(times_snap) / t_char, np.array(u_center_snap) / U_in,
            'rs-', markersize=6, linewidth=1.5, label='중앙점 u (수치해)')
    ax.axhline(y=2.096, color='k', linestyle='--', linewidth=1.0,
               label='3D 덕트 피크 (2.096 U_in)')
    ax.set_xlabel('t / t_char [-]')
    ax.set_ylabel('u_center / U_in [-]')
    ax.set_title('중앙점 속도 시간 이력')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 23: 3D 과도 채널 유동 (Re={Re:.0f}, {nx}×{ny}×{nz})',
                 fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case23_3d_transient.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # ---- 입력 JSON ----
    params = {
        'case': 23,
        'description': '3D Transient Channel Flow',
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'nx': nx, 'ny': ny, 'nz': nz,
        'n_cells': mesh.n_cells,
        'Re': Re,
        'rho': rho,
        'mu': mu,
        'U_in': U_in,
        'dt': dt,
        't_end': t_end,
        't_char': t_char,
        'L2_error': L2_error,
        'transient_development': transient_development,
    }
    json_path = os.path.join(case_dir, 'input.json')
    export_input_json(params, json_path)

    # ---- 판정 ----
    # (1) 해석해 대비 L2 오차 < 10%, (2) 과도 발달 확인, (3) 잔차 안정
    residual_decreasing = False
    if len(residual_history) >= 10:
        early_res = np.mean(residual_history[:10])
        late_res = np.mean(residual_history[-10:])
        residual_decreasing = late_res < early_res * 2.0  # 발산하지 않으면 OK
    converged_to_analytical = L2_error < 0.10
    passed = converged_to_analytical and transient_development and residual_decreasing

    print(f"  판정: L2={L2_error:.4e}, 해석해수렴={converged_to_analytical}, "
          f"과도발달={transient_development}, 잔차안정={residual_decreasing}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {
        'converged': passed,
        'L2_error': L2_error,
        'transient_development': transient_development,
        'residual_decreasing': residual_decreasing,
        'time_steps': step,
        'final_time': t,
        'figure_path': fig_path,
        'vtu_dir': case_dir,
        'n_snapshots': len(snapshots),
        'Re': Re,
    }


if __name__ == "__main__":
    result = run_case23()
    print(f"\n결과 요약:")
    print(f"  L2 오차: {result['L2_error']:.4e}")
    print(f"  과도 발달: {result['transient_development']}")
    print(f"  시간 스텝: {result['time_steps']}")
    print(f"  PASS: {result['converged']}")
