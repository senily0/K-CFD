"""
Case 12: AMR 적응 격자 세분화 검증.

계단 소스가 있는 확산 문제에서 AMR 적용.
기울기 큰 영역에서 자동 세분화 확인.
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
from mesh.amr import AMRMesh, GradientJumpEstimator, AMRSolverLoop
from core.fields import ScalarField
from core.fvm_operators import FVMSystem, diffusion_operator, source_term
from core.linear_solver import solve_linear_system
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case12(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    AMR 적응 격자 검증.

    Returns
    -------
    result : {'n_cells_initial': int, 'n_cells_final': int,
              'n_refinements': int, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 12: AMR Adaptive Mesh Refinement")
    print("=" * 60)

    # 파라미터
    Lx = 1.0
    Ly = 1.0
    nx_init = 10
    ny_init = 10
    gamma_val = 1.0   # 확산 계수
    n_amr_cycles = 3   # AMR 반복 횟수

    # 초기 격자
    base_mesh = generate_channel_mesh(Lx, Ly, nx_init, ny_init)
    print(f"  초기 격자: {base_mesh.summary()}")

    # AMR 메쉬
    amr = AMRMesh(base_mesh, max_level=3)

    cell_counts = [base_mesh.n_cells]
    error_history = []

    for cycle in range(n_amr_cycles):
        print(f"\n  --- AMR Cycle {cycle + 1} ---")

        # 현재 활성 메쉬
        if cycle == 0:
            mesh = base_mesh
        else:
            mesh = amr.get_active_mesh()

        n_cells = mesh.n_cells
        print(f"  셀 수: {n_cells}")

        # 확산 문제 풀기: -div(gamma*grad(T)) = S
        # 소스: 계단 함수 S = 100 (x>0.4 and x<0.6 and y>0.4 and y<0.6)
        T = ScalarField(mesh, "T")
        T.set_uniform(0.0)

        # 경계조건: 모든 벽 T=0 (Dirichlet)
        for bname in mesh.boundary_patches:
            fids = mesh.boundary_patches[bname]
            T.boundary_values[bname] = np.zeros(len(fids))

        # 소스항
        S = np.zeros(n_cells)
        for ci in range(n_cells):
            cc = mesh.cells[ci].center
            if 0.4 < cc[0] < 0.6 and 0.4 < cc[1] < 0.6:
                S[ci] = 100.0

        # FVM 시스템
        gamma = ScalarField(mesh, "gamma")
        gamma.values[:] = gamma_val

        system = FVMSystem(n_cells)
        diffusion_operator(mesh, gamma, system)
        source_term(mesh, S, system)

        # 경계 적용 (Dirichlet T=0)
        for bname, fids in mesh.boundary_patches.items():
            for local_idx, fid in enumerate(fids):
                face = mesh.faces[fid]
                owner = face.owner
                d_Pf = np.linalg.norm(face.center - mesh.cells[owner].center)
                if d_Pf < 1e-30:
                    continue
                coeff = gamma_val * face.area / d_Pf
                system.add_source(owner, coeff * 0.0)  # T_wall = 0

        # 풀기
        T.values = solve_linear_system(system, T.values, method='direct')

        # 오차 추정
        error = GradientJumpEstimator.estimate(mesh, T)
        max_error = float(np.max(error))
        mean_error = float(np.mean(error))
        error_history.append({'max': max_error, 'mean': mean_error})
        print(f"  오차 지표: max={max_error:.4e}, mean={mean_error:.4e}")

        # AMR: 상위 30% 셀 세분화
        if cycle < n_amr_cycles - 1:
            threshold = np.percentile(error, 70)
            active_ids = amr.get_active_cells()
            cells_to_refine = []
            for local_idx, amr_id in enumerate(active_ids):
                if local_idx < len(error) and error[local_idx] > threshold:
                    cells_to_refine.append(amr_id)

            print(f"  세분화 셀 수: {len(cells_to_refine)}")
            amr.refine_cells(cells_to_refine)

            new_mesh = amr.get_active_mesh()
            cell_counts.append(new_mesh.n_cells)

    # 최종 격자 정보
    final_mesh = amr.get_active_mesh() if n_amr_cycles > 1 else base_mesh
    n_final = final_mesh.n_cells
    print(f"\n  최종 셀 수: {n_final} (초기: {nx_init * ny_init})")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) 초기 온도 분포 (마지막 풀이)
    ax = axes[0]
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(mesh.n_cells)])
    y_cells = np.array([mesh.cells[ci].center[1] for ci in range(mesh.n_cells)])
    scatter = ax.scatter(x_cells, y_cells, c=T.values, cmap='hot',
                         s=20, alpha=0.8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Temperature Field')
    plt.colorbar(scatter, ax=ax)
    ax.set_aspect('equal')

    # 2) 오차 분포
    ax = axes[1]
    scatter2 = ax.scatter(x_cells, y_cells, c=error, cmap='YlOrRd',
                          s=20, alpha=0.8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Error Indicator')
    plt.colorbar(scatter2, ax=ax)
    ax.set_aspect('equal')

    # 3) 셀 수 이력
    ax = axes[2]
    ax.bar(range(len(cell_counts)), cell_counts, color='steelblue')
    ax.set_xlabel('AMR Cycle')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Mesh Adaptation History')
    ax.set_xticks(range(len(cell_counts)))

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case12_amr.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- VTU / JSON export (final active mesh) ---
    case_dir = os.path.join(results_dir, 'case12')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'nx': nx_init, 'ny': ny_init,
              'max_level': 3, 'n_cycles': n_amr_cycles,
              'gamma': gamma_val, 'refine_fraction': 0.3}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {'temperature': T.values}
        export_mesh_to_vtu(final_mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case12 export skipped: {e}")

    return {
        'n_cells_initial': nx_init * ny_init,
        'n_cells_final': n_final,
        'n_refinements': n_amr_cycles,
        'cell_counts': cell_counts,
        'error_history': error_history,
        'figure_path': fig_path,
    }


if __name__ == "__main__":
    result = run_case12()
    print(f"\n  결과: {result['n_cells_initial']} → {result['n_cells_final']} cells")
    if result['n_cells_final'] > result['n_cells_initial']:
        print("  V AMR 세분화 동작 확인")
    else:
        print("  X AMR 세분화 미작동")
