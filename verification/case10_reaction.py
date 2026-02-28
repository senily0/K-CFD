"""
Case 10: 플러그 흐름 반응기 (1차 반응 검증).

A -> B, 반응률 R_A = -k_r * C_A
해석해: C_A(x) = C_A0 * exp(-k_r * x / u)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.mesh_generator import generate_channel_mesh
from core.fields import ScalarField, VectorField
from core.interpolation import compute_mass_flux
from models.chemistry import FirstOrderReaction, SpeciesTransportSolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case10(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    플러그 흐름 반응기 검증.

    Returns
    -------
    result : {'L2_error': float, 'converged': bool, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 10: Plug Flow Reactor (1st Order Reaction)")
    print("=" * 60)

    # 파라미터
    Lx = 1.0
    Ly = 0.02
    nx = 100
    ny = 1
    rho = 1.0
    u_in = 1.0
    k_r = 2.0       # 반응 속도 상수
    D = 1e-6         # 매우 작은 확산 (대류 지배)
    C_A0 = 1.0       # 입구 농도

    # 격자
    mesh = generate_channel_mesh(Lx, Ly, nx, ny)
    print(f"  {mesh.summary()}")

    # 속도장 (균일)
    U = VectorField(mesh, "U")
    U.values[:, 0] = u_in
    for bname in mesh.boundary_patches:
        fids = mesh.boundary_patches[bname]
        U.boundary_values[bname] = np.zeros((len(fids), mesh.ndim))
        U.boundary_values[bname][:, 0] = u_in

    # 질량유속
    mass_flux = compute_mass_flux(U, rho, mesh)

    # 반응 모델
    reaction = FirstOrderReaction(k_r=k_r)

    # 종 수송 솔버
    solver = SpeciesTransportSolver(mesh, rho=rho, D=D, reaction=reaction)
    solver.C.set_uniform(C_A0)

    # 경계조건
    solver.set_bc('inlet', 'dirichlet', C_A0)
    solver.set_bc('outlet', 'zero_gradient')
    solver.set_bc('wall_bottom', 'zero_gradient')
    solver.set_bc('wall_top', 'zero_gradient')

    # 풀기
    print("  해석 중...")
    result = solver.solve_steady(U, mass_flux, max_iter=300, tol=1e-8)
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # 셀 x 좌표 및 수치해
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(mesh.n_cells)])
    C_num = solver.C.values.copy()

    # 해석해
    C_anal = C_A0 * np.exp(-k_r * x_cells / u_in)

    # L2 오차
    L2_error = np.sqrt(np.mean((C_num - C_anal) ** 2)) / max(np.max(C_anal), 1e-15)
    print(f"  L2 상대오차: {L2_error:.4e}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x_fine = np.linspace(0, Lx, 500)
    C_fine = C_A0 * np.exp(-k_r * x_fine / u_in)
    ax.plot(x_fine, C_fine, 'r-', linewidth=2, label='Analytical')
    ax.plot(x_cells, C_num, 'bo', markersize=3, label='Numerical (FVM)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('C_A / C_A0')
    ax.set_title(f'Plug Flow Reactor (k_r={k_r}, u={u_in})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 잔차
    ax = axes[1]
    if result['residuals']:
        ax.semilogy(result['residuals'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case10_reaction.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case10')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'nx': nx, 'ny': ny,
              'u': u_in, 'k_r': k_r, 'D': D, 'C_A0': C_A0}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {'concentration_A': solver.C.values}
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case10 export skipped: {e}")

    return {
        'L2_error': L2_error,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'figure_path': fig_path,
        'residuals': result['residuals'],
    }


if __name__ == "__main__":
    result = run_case10()
    print(f"\n  결과: L2 오차 = {result['L2_error']:.4e}")
    if result['L2_error'] < 0.05:
        print("  V 검증 통과 (L2 < 5%)")
    else:
        print("  X 검증 실패")
