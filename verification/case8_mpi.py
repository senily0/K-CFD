"""
Case 8: MPI 병렬화 검증.

1, 2, 4 파티션에서 Poiseuille 풀이 비교.
결과 일치 확인 (고스트 교환 정확도).
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
from parallel.mpi_solver import MPISIMPLESolver
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case8(results_dir: str = "results",
              figures_dir: str = "figures") -> dict:
    """
    MPI 병렬 Poiseuille 검증.

    Returns
    -------
    result : {'max_partition_diff': float, 'converged': bool, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 8: MPI Parallel Poiseuille Verification")
    print("=" * 60)

    # 파라미터
    Lx = 1.0
    Ly = 0.1
    nx = 40
    ny = 10
    rho = 1.0
    mu = 0.01
    dpdx = -1.0

    # 격자
    mesh = generate_channel_mesh(Lx, Ly, nx, ny)
    print(f"  {mesh.summary()}")

    # 1, 2, 4 파티션 비교
    results_by_nparts = {}
    for n_parts in [1, 2, 4]:
        print(f"\n  --- {n_parts} partitions ---")
        mpi_solver = MPISIMPLESolver(mesh, rho=rho, mu=mu, n_parts=n_parts)
        result = mpi_solver.solve_parallel_poiseuille(
            dpdx=dpdx, max_iter=500, tol=1e-4)

        print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")
        print(f"  고스트 교환 오차: {result['exchange_error']:.2e}")
        for rank, info in result['partition_info'].items():
            print(f"    Part {rank}: {info['n_local']} cells, "
                  f"{info['n_ghost']} ghosts, neighbors={info['neighbors']}")

        results_by_nparts[n_parts] = result

    # 결과 비교: 1파티션 대비 차이
    ref_u = results_by_nparts[1]['u_values']
    max_diffs = {}
    for n_parts in [2, 4]:
        u_test = results_by_nparts[n_parts]['u_values']
        max_diff = np.max(np.abs(u_test - ref_u))
        max_diffs[n_parts] = max_diff
        print(f"\n  {n_parts} partitions vs 1: max|u_diff| = {max_diff:.2e}")

    max_partition_diff = max(max_diffs.values()) if max_diffs else 0.0

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 파티션 시각화 (4파티션)
    ax = axes[0]
    part_ids = MPISIMPLESolver(mesh, rho=rho, mu=mu,
                                n_parts=4).part_ids
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(mesh.n_cells)])
    y_cells = np.array([mesh.cells[ci].center[1] for ci in range(mesh.n_cells)])
    scatter = ax.scatter(x_cells, y_cells, c=part_ids, cmap='Set1',
                         s=10, alpha=0.8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Domain Partitioning (4 parts, RCB)')
    plt.colorbar(scatter, ax=ax, label='Partition ID')

    # 속도 프로파일 비교
    ax = axes[1]
    # 출구 근처 y-프로파일
    x_target = Lx * 0.8
    tol_x = Lx / nx * 1.5

    for n_parts, style in [(1, 'r-'), (2, 'g--'), (4, 'b:')]:
        u_vals = results_by_nparts[n_parts]['u_values']
        y_prof, u_prof = [], []
        for ci in range(mesh.n_cells):
            if abs(x_cells[ci] - x_target) < tol_x:
                y_prof.append(y_cells[ci])
                u_prof.append(u_vals[ci, 0])
        idx = np.argsort(y_prof)
        y_prof = np.array(y_prof)[idx]
        u_prof = np.array(u_prof)[idx]
        ax.plot(u_prof, y_prof, style, linewidth=2, label=f'{n_parts} part(s)')

    ax.set_xlabel('u [m/s]')
    ax.set_ylabel('y [m]')
    ax.set_title('Velocity Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case8_mpi.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {fig_path}")

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case8')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'nx': nx, 'ny': ny,
              'n_parts_list': [1, 2, 4], 'rho': rho, 'mu': mu, 'dpdx': dpdx}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        ref_result = results_by_nparts[1]
        partition_ids_export = MPISIMPLESolver(mesh, rho=rho, mu=mu, n_parts=4).part_ids
        cell_data = {
            'velocity_x': ref_result['u_values'][:, 0],
            'pressure': ref_result['u_values'][:, 0] * 0.0,
            'partition_id': partition_ids_export.astype(float),
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case8 export skipped: {e}")

    return {
        'max_partition_diff': max_partition_diff,
        'converged': all(r['converged'] for r in results_by_nparts.values()),
        'exchange_errors': {k: v['exchange_error']
                           for k, v in results_by_nparts.items()},
        'figure_path': fig_path,
    }


if __name__ == "__main__":
    result = run_case8()
    print(f"\n  결과: 최대 파티션 차이 = {result['max_partition_diff']:.2e}")
    if result['max_partition_diff'] < 1e-6:
        print("  V 검증 통과 (차이 < 1e-6)")
    else:
        print("  X 검증 실패")
