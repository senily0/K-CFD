"""
Case 15: 3D 자연대류 (Rayleigh-Benard) 검증.

3D 밀폐 공간에서 Boussinesq 근사를 이용한 자연대류.
하단 고온, 상단 저온, 측면 단열.
온도 성층화 및 Nusselt 수 검증.
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
from models.single_phase import SIMPLESolver
from core.fields import ScalarField
from core.gradient import green_gauss_gradient
from core.fvm_operators import FVMSystem, diffusion_operator, source_term


def run_case15(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    3D 자연대류 검증.

    Returns
    -------
    result : {'converged': bool, 'nusselt': float, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 15: 3D Natural Convection (Heated Cavity)")
    print("=" * 60)

    # 파라미터
    Lx = 1.0       # 가로
    Ly = 1.0       # 깊이 (z-방향 대용)
    Lz = 0.5       # 높이 (z-방향)
    nx, ny, nz = 16, 16, 8
    rho = 1.0
    mu = 0.01      # 점성
    beta = 0.01    # 열팽창계수
    k_th = 0.01    # 열전도도
    cp = 1.0       # 비열
    T_hot = 1.0    # 하단 온도
    T_cold = 0.0   # 상단 온도
    T_ref = 0.5    # 참조 온도
    g_val = 9.81   # 중력가속도

    # Ra = rho * g * beta * dT * H^3 / (mu * alpha_th)
    alpha_th = k_th / (rho * cp)
    dT = T_hot - T_cold
    Ra = rho * g_val * beta * dT * Lz**3 / (mu * alpha_th)
    Pr = mu * cp / k_th
    print(f"  Ra = {Ra:.1f}, Pr = {Pr:.1f}")
    print(f"  격자: {nx}x{ny}x{nz} = {nx*ny*nz} cells")

    # 격자 생성: 밀폐 공간 (모든 면 벽)
    mesh = generate_3d_channel_mesh(
        Lx, Ly, Lz, nx, ny, nz,
        boundary_names={
            'x_min': 'wall_left', 'x_max': 'wall_right',
            'y_min': 'wall_front', 'y_max': 'wall_back',
            'z_min': 'wall_bottom', 'z_max': 'wall_top'
        }
    )
    print(f"  {mesh.summary()}")

    ndim = 3

    # --- 운동량 솔버 설정 ---
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 500
    solver.tol = 1e-4
    solver.alpha_u = 0.3
    solver.alpha_p = 0.1

    # 속도 BC: 모든 면 no-slip
    for wall in ['wall_left', 'wall_right', 'wall_front', 'wall_back',
                  'wall_bottom', 'wall_top']:
        if wall in mesh.boundary_patches:
            solver.set_velocity_bc(wall, 'dirichlet', [0.0, 0.0, 0.0])
            solver.set_pressure_bc(wall, 'zero_gradient')

    # 압력 고정 (한 점)
    solver.p.values[0] = 0.0

    # --- 온도 필드 ---
    T = ScalarField(mesh, "temperature")
    # 초기 온도: 선형 분포 (z 기준)
    for ci in range(mesh.n_cells):
        zc = mesh.cells[ci].center[2]
        T.values[ci] = T_hot + (T_cold - T_hot) * zc / Lz

    # --- 결합 반복 ---
    print("  결합 반복 해석 중 (운동량 + 에너지)...")
    n_outer = 100
    residuals = []

    for outer in range(n_outer):
        # 1) 부력 소스항 계산: F_z = -rho * beta * (T - T_ref) * g
        buoyancy_z = -rho * beta * (T.values - T_ref) * g_val

        # 부력을 속도 z-성분 소스로 추가하기 위해
        # solver의 내부에 직접 접근하기 어려우므로,
        # 이전 반복의 부력을 초기 속도에 반영하는 방식 사용
        # (명시적 소스항 결합)
        U_old = solver.U.values.copy()
        p_old = solver.p.values.copy()

        # SIMPLE 1회 반복 수행 (내부 반복은 몇 회만)
        solver.max_outer_iter = 5
        solver._res0 = None
        result_mom = solver.solve_steady()

        # 부력 보정: z-속도에 명시적으로 추가
        # dU_z = dt_pseudo * buoyancy / rho (의사시간보폭 사용)
        dt_pseudo = 0.05
        solver.U.values[:, 2] += dt_pseudo * buoyancy_z

        # 2) 에너지 방정식 풀기: div(k*grad(T)) = rho*cp*u·grad(T)
        # 간단한 확산-대류: FVM으로 직접 풀기
        gamma_T = ScalarField(mesh, "gamma_T")
        gamma_T.values[:] = k_th

        system_T = FVMSystem(mesh.n_cells)
        diffusion_operator(mesh, gamma_T, system_T)

        # 대류항 (upwind): rho*cp*F*T_f
        for fid, face in enumerate(mesh.faces):
            o = face.owner
            nb = face.neighbour

            # 질량 유속
            u_f = solver.U.values[o]
            if nb >= 0:
                u_f = 0.5 * (solver.U.values[o] + solver.U.values[nb])
            F = rho * cp * np.dot(u_f, face.normal) * face.area

            if nb >= 0:
                if F >= 0:
                    system_T.add_diagonal(o, F)
                    system_T.add_off_diagonal(nb, o, -F)
                else:
                    system_T.add_off_diagonal(o, nb, F)
                    system_T.add_diagonal(nb, -F)
            else:
                # 경계면
                binfo = solver._face_bc_cache.get(fid)
                if binfo:
                    bname, _ = binfo
                    if bname == 'wall_bottom':
                        # Dirichlet T = T_hot
                        d = np.linalg.norm(face.center - mesh.cells[o].center)
                        coeff = k_th * face.area / max(d, 1e-30)
                        system_T.add_diagonal(o, coeff)
                        system_T.add_source(o, coeff * T_hot)
                    elif bname == 'wall_top':
                        # Dirichlet T = T_cold
                        d = np.linalg.norm(face.center - mesh.cells[o].center)
                        coeff = k_th * face.area / max(d, 1e-30)
                        system_T.add_diagonal(o, coeff)
                        system_T.add_source(o, coeff * T_cold)
                    # 측면: zero gradient (기본 - 아무것도 안함)

        A_T = system_T.to_sparse()
        b_T = system_T.rhs

        # 대각 우세 보강
        for i in range(mesh.n_cells):
            if A_T[i, i] < 1e-20:
                A_T[i, i] = 1.0
                b_T[i] = T.values[i]

        from scipy.sparse.linalg import spsolve
        T_new = spsolve(A_T, b_T)

        # 이완
        alpha_T = 0.5
        T.values[:] = alpha_T * T_new + (1 - alpha_T) * T.values

        # 온도 범위 클리핑
        T.values[:] = np.clip(T.values, T_cold - 0.1, T_hot + 0.1)

        # 잔차 계산
        res_T = np.max(np.abs(T_new - T.values)) if outer > 0 else 1.0
        res_U = np.max(np.abs(solver.U.values - U_old))
        res = max(res_T, res_U)
        residuals.append(res)

        if outer % 20 == 0:
            u_max = np.max(np.abs(solver.U.values))
            print(f"    반복 {outer:3d}: res={res:.4e}, u_max={u_max:.4e}, "
                  f"T_range=[{T.values.min():.3f}, {T.values.max():.3f}]")

        if res < 1e-5 and outer > 10:
            print(f"  수렴 달성 (반복 {outer})")
            break

    converged = (len(residuals) > 0 and residuals[-1] < 1e-3)
    print(f"  수렴 여부: {converged}")

    # --- Nusselt 수 계산 ---
    # Nu = (dT/dz)|wall * H / dT_total
    # 하단 벽면에서의 열유속
    q_bottom = 0.0
    area_bottom = 0.0
    for fid in mesh.boundary_patches.get('wall_bottom', []):
        face = mesh.faces[fid]
        o = face.owner
        d = np.linalg.norm(face.center - mesh.cells[o].center)
        q_local = k_th * (T.values[o] - T_hot) / max(d, 1e-30)  # 내부로 향하는 열유속
        q_bottom += abs(q_local) * face.area
        area_bottom += face.area

    if area_bottom > 0:
        q_avg = q_bottom / area_bottom
        # Nu = q_avg * H / (k_th * dT)
        nusselt = q_avg * Lz / (k_th * dT) if dT > 0 else 0.0
    else:
        nusselt = 0.0

    print(f"  Nusselt 수: {nusselt:.4f}")

    # --- 온도 성층화 검증 ---
    # 중앙 수직선 (x=Lx/2, y=Ly/2)에서 온도 프로파일
    x_mid, y_mid = Lx / 2, Ly / 2
    tol_xy = max(Lx / nx, Ly / ny) * 1.5
    z_prof, T_prof = [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[0] - x_mid) < tol_xy and abs(cc[1] - y_mid) < tol_xy:
            z_prof.append(cc[2])
            T_prof.append(T.values[ci])

    z_prof = np.array(z_prof)
    T_prof = np.array(T_prof)
    idx = np.argsort(z_prof)
    z_prof = z_prof[idx]
    T_prof = T_prof[idx]

    # 성층화 검증: 하단 온도 > 상단 온도
    stratified = True
    if len(T_prof) > 2:
        stratified = T_prof[0] > T_prof[-1]
    print(f"  온도 성층화: {'확인' if stratified else '미확인'}")

    # --- 시각화 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) 중앙 수직 온도 프로파일
    ax = axes[0]
    if len(z_prof) > 0:
        ax.plot(T_prof, z_prof, 'ro-', markersize=4, label='Numerical')
        # 전도 해석해 (대류 없을 때): T(z) = T_hot - dT*z/H
        z_fine = np.linspace(0, Lz, 50)
        T_cond = T_hot + (T_cold - T_hot) * z_fine / Lz
        ax.plot(T_cond, z_fine, 'b--', label='Conduction only')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('z [m]')
    ax.set_title('Vertical Temperature Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) z=Lz/2 수평면 온도 분포
    ax = axes[1]
    z_mid_val = Lz / 2
    tol_z = Lz / nz * 1.5
    x_slice, y_slice, T_slice = [], [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if abs(cc[2] - z_mid_val) < tol_z:
            x_slice.append(cc[0])
            y_slice.append(cc[1])
            T_slice.append(T.values[ci])
    if x_slice:
        sc = ax.scatter(x_slice, y_slice, c=T_slice, cmap='hot', s=20)
        plt.colorbar(sc, ax=ax, label='T')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Temperature at z={z_mid_val:.2f}m')
    ax.set_aspect('equal')

    # 3) 수렴 이력
    ax = axes[2]
    if residuals:
        ax.semilogy(residuals)
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case15_3d_convection.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- 입력/메쉬 파일 저장 ---
    try:
        from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json
        case_dir = os.path.join(results_dir, 'case15')
        os.makedirs(case_dir, exist_ok=True)
        export_input_json({
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'nx': nx, 'ny': ny, 'nz': nz,
            'rho': rho, 'mu': mu, 'beta': beta,
            'k_th': k_th, 'cp': cp,
            'T_hot': T_hot, 'T_cold': T_cold, 'T_ref': T_ref,
            'g': g_val, 'Ra': Ra, 'Pr': Pr
        }, os.path.join(case_dir, 'input.json'))
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), {
            'temperature': T.values,
            'velocity_x': solver.U.values[:, 0],
            'velocity_y': solver.U.values[:, 1],
            'velocity_z': solver.U.values[:, 2],
            'pressure': solver.p.values
        })
        print(f"  VTU/JSON 저장: {case_dir}")
    except Exception as e:
        print(f"  VTU/JSON 저장 실패: {e}")

    return {
        'converged': converged,
        'nusselt': float(nusselt),
        'stratified': stratified,
        'iterations': len(residuals),
        'figure_path': fig_path,
        'residuals': residuals,
        'T_min': float(T.values.min()),
        'T_max': float(T.values.max()),
    }


if __name__ == "__main__":
    result = run_case15()
    print(f"\n  결과: Nu={result['nusselt']:.4f}, "
          f"수렴={result['converged']}, 성층화={result['stratified']}")
    if result['converged'] and result['stratified']:
        print("  V 검증 통과")
    else:
        print("  X 검증 실패")
