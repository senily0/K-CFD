"""
Case 14: 3D 뚜껑 구동 공동(Lid-Driven Cavity) 유동 검증.

3D 정육면체 공동, 상부 벽(y=Ly) 이동 (U_lid=1.0 m/s, x방향).
Re = 100, ρ=1.0, μ=0.01
격자: 16×16×16 셀 (4096 cells)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator_3d import generate_3d_cavity_mesh
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json
from models.single_phase import SIMPLESolver


def run_case14(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    3D 뚜껑 구동 공동 유동 검증 실행.

    Returns
    -------
    result : {'converged': bool, 'iterations': int,
              'figure_path': str, 'residuals': list}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 14: 3D Lid-Driven Cavity 유동 검증")
    print("=" * 60)

    # 파라미터
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    nx, ny, nz = 16, 16, 16
    rho = 1.0
    U_lid = 1.0
    Re = 100
    mu = rho * U_lid * Lx / Re   # = 0.01

    print(f"  Re = {Re}, rho = {rho}, mu = {mu:.4f}, U_lid = {U_lid}")

    # 격자 생성
    print("  3D 격자 생성 중...")
    mesh = generate_3d_cavity_mesh(Lx, Ly, Lz, nx, ny, nz)
    print(f"  {mesh.summary()}")

    # 솔버 설정
    solver = SIMPLESolver(mesh, rho=rho, mu=mu)
    solver.max_outer_iter = 1000
    solver.tol = 1e-4
    solver.alpha_u = 0.5
    solver.alpha_p = 0.2

    # 경계조건 — 속도
    solver.set_velocity_bc('lid',          'dirichlet', [U_lid, 0.0, 0.0])
    solver.set_velocity_bc('wall_bottom',  'dirichlet', [0.0,   0.0, 0.0])
    solver.set_velocity_bc('wall_left',    'dirichlet', [0.0,   0.0, 0.0])
    solver.set_velocity_bc('wall_right',   'dirichlet', [0.0,   0.0, 0.0])
    solver.set_velocity_bc('wall_front',   'dirichlet', [0.0,   0.0, 0.0])
    solver.set_velocity_bc('wall_back',    'dirichlet', [0.0,   0.0, 0.0])

    # 경계조건 — 압력 (모든 벽 zero_gradient, 참조 압력은 솔버 내부 고정)
    for patch in ['lid', 'wall_bottom', 'wall_left',
                  'wall_right', 'wall_front', 'wall_back']:
        solver.set_pressure_bc(patch, 'zero_gradient')

    # 초기 조건: u = 0
    solver.U.values[:] = 0.0
    solver.p.values[:] = 0.0

    # 해석 실행
    print("  해석 중...")
    result = solver.solve_steady()
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # ---- 중앙 수직선(x=0.5, z=0.5) u-속도 프로파일 추출 ----
    x_mid = Lx / 2.0
    z_mid = Lz / 2.0
    tol_xz = min(Lx / nx, Lz / nz) * 1.5

    y_prof, u_prof = [], []
    for ci in range(mesh.n_cells):
        cc = mesh.cells[ci].center
        if (abs(cc[0] - x_mid) < tol_xz and
                abs(cc[2] - z_mid) < tol_xz):
            y_prof.append(cc[1])
            u_prof.append(solver.U.values[ci, 0])

    y_prof = np.array(y_prof)
    u_prof = np.array(u_prof)
    if len(y_prof) > 0:
        idx = np.argsort(y_prof)
        y_prof = y_prof[idx]
        u_prof = u_prof[idx]

    u_max_lid_region = float(np.max(u_prof)) if len(u_prof) > 0 else 0.0
    print(f"  중앙선 u_max: {u_max_lid_region:.4f} (기대: ~{U_lid:.1f} 근방)")

    # ---- 시각화 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 서브플롯 1: 중앙선 u-속도 프로파일
    ax = axes[0]
    if len(y_prof) > 0:
        ax.plot(u_prof, y_prof, 'b-o', linewidth=1.5, markersize=3,
                label='수치해 (FVM, 3D)')
        # lid 속도 참조선
        ax.axhline(y=Ly, color='r', linestyle='--', linewidth=1.0,
                   label=f'Lid (y={Ly})')
        ax.axvline(x=0.0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'3D Cavity 중앙선 u-속도 (x=0.5, z=0.5)\nRe={Re}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, Ly)

    # 서브플롯 2: 수렴 이력
    ax = axes[1]
    if result['residuals']:
        ax.semilogy(result['residuals'], 'b-', linewidth=1.2)
        ax.axhline(y=solver.tol, color='r', linestyle='--',
                   linewidth=1.0, label=f'허용 잔차 ({solver.tol:.0e})')
    ax.set_xlabel('반복 횟수')
    ax.set_ylabel('정규화 잔차')
    ax.set_title('SIMPLE 수렴 이력')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Case 14: 3D Lid-Driven Cavity (Re={Re}, {nx}x{ny}x{nz})',
                 fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case14_3d_cavity.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # ---- VTU 내보내기 ----
    vtu_path = os.path.join(results_dir, 'case14_3d_cavity.vtu')
    cell_data = {
        'u': solver.U.values[:, 0],
        'v': solver.U.values[:, 1],
        'w': solver.U.values[:, 2],
        'p': solver.p.values,
        'velocity_magnitude': np.linalg.norm(solver.U.values, axis=1),
    }
    export_mesh_to_vtu(mesh, vtu_path, cell_data=cell_data)
    print(f"  VTU 저장: {vtu_path}")

    # ---- 입력 JSON 저장 ----
    params = {
        'case': 14,
        'description': '3D Lid-Driven Cavity',
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'nx': nx, 'ny': ny, 'nz': nz,
        'n_cells': mesh.n_cells,
        'Re': Re,
        'rho': rho,
        'mu': mu,
        'U_lid': U_lid,
        'alpha_u': solver.alpha_u,
        'alpha_p': solver.alpha_p,
        'max_iter': solver.max_outer_iter,
        'tol': solver.tol,
        'converged': result['converged'],
        'iterations': result['iterations'],
    }
    json_path = os.path.join(results_dir, 'case14_input.json')
    export_input_json(params, json_path)
    print(f"  입력 JSON 저장: {json_path}")

    # ---- 물리적 타당성 검사 ----
    # 뚜껑 근처(y > 0.8)에서 u는 양수, 중앙(y~0.5)에서 음수여야 함
    physically_reasonable = False
    if len(u_prof) >= 4:
        u_top = float(np.mean(u_prof[y_prof > 0.7]))
        u_mid = float(np.mean(u_prof[(y_prof > 0.3) & (y_prof < 0.6)]))
        sign_change = (u_top > 0) and (u_mid < 0)
        physically_reasonable = sign_change
        print(f"  상부 평균 u: {u_top:.4f}, 중앙 평균 u: {u_mid:.4f}")
        print(f"  부호 변화(물리적 타당성): {sign_change}")

    result_data = {
        'converged': result['converged'],
        'iterations': result['iterations'],
        'figure_path': fig_path,
        'residuals': result['residuals'],
        'y_profile': y_prof.tolist() if len(y_prof) > 0 else [],
        'u_profile': u_prof.tolist() if len(u_prof) > 0 else [],
        'u_max': u_max_lid_region,
        'physically_reasonable': physically_reasonable,
    }

    return result_data


if __name__ == "__main__":
    result = run_case14()
    print("\n  결과 요약:")
    print(f"    수렴: {result['converged']}")
    print(f"    반복 횟수: {result['iterations']}")
    print(f"    u_max (중앙선): {result['u_max']:.4f}")
    print(f"    물리적 타당성: {result['physically_reasonable']}")
    if result['converged'] and result['physically_reasonable']:
        print("  검증 통과")
    else:
        print("  검증 미완료 — 수렴 또는 물리적 타당성 확인 필요")
