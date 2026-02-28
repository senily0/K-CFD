"""
Case 16: 전처리기(Preconditioner) 비교 검증.

2D Laplace 방정식 ∇²T = 0 (T=0 하단, T=1 상단) 을 조립하고
none / jacobi / ilu0 / iluk / amg 전처리기별 반복 횟수·시간·오차를 비교.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.sparse.linalg import bicgstab as scipy_bicgstab
from core.preconditioner import create_preconditioner, HAS_PYAMG
from core.fvm_operators import FVMSystem
from mesh.mesh_generator import generate_channel_mesh
from mesh.vtk_exporter import export_mesh_to_vtu


def _build_convdiff_system(mesh, gamma_val=0.01, u_conv=1.0):
    """
    2D 대류-확산 방정식 조립: u·∂T/∂x - γ∇²T = 0

    고 Peclet 수(Pe = u*L/γ = 100)로 나쁜 조건수를 만들어
    전처리기 효과가 뚜렷하게 나타나도록 한다.

    경계조건:
      inlet      : T = 1  (Dirichlet)
      outlet     : zero_gradient
      wall_bottom: T = 0  (Dirichlet)
      wall_top   : T = 0  (Dirichlet)
    """
    n = mesh.n_cells
    system = FVMSystem(n)

    dirichlet_patches = {
        'inlet': 1.0,
        'wall_bottom': 0.0,
        'wall_top': 0.0,
    }
    dirichlet_face_value = {}
    for bname, phi_b in dirichlet_patches.items():
        for fid in mesh.boundary_patches.get(bname, []):
            dirichlet_face_value[fid] = phi_b

    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        nx_dir = face.normal[0] if hasattr(face, 'normal') else 0.0

        if face.neighbour >= 0:
            neighbour = face.neighbour
            xO = mesh.cells[owner].center
            xN = mesh.cells[neighbour].center
            d_PN = float(np.linalg.norm(xN - xO))
            if d_PN < 1e-30:
                continue

            # 확산
            diff_coeff = gamma_val * face.area / d_PN
            system.add_diagonal(owner, diff_coeff)
            system.add_diagonal(neighbour, diff_coeff)
            system.add_off_diagonal(owner, neighbour, -diff_coeff)
            system.add_off_diagonal(neighbour, owner, -diff_coeff)

            # 대류 (upwind): F = u_conv * face.area * sign(n_x)
            # face normal points from owner to neighbour
            F = u_conv * face.area * nx_dir
            system.add_diagonal(owner, max(F, 0.0))
            system.add_off_diagonal(owner, neighbour, min(F, 0.0))
            system.add_diagonal(neighbour, max(-F, 0.0))
            system.add_off_diagonal(neighbour, owner, min(-F, 0.0))

        elif fid in dirichlet_face_value:
            xO = mesh.cells[owner].center
            d_Pf = float(np.linalg.norm(face.center - xO))
            if d_Pf < 1e-30:
                continue
            diff_coeff = gamma_val * face.area / d_Pf
            phi_b = dirichlet_face_value[fid]
            system.add_diagonal(owner, diff_coeff)
            system.add_source(owner, diff_coeff * phi_b)
            # 대류 경계 (inlet: F<0 means inflow to owner)
            F = u_conv * face.area * nx_dir
            if F >= 0:
                system.add_diagonal(owner, F)
            else:
                system.add_source(owner, -F * phi_b)
        else:
            # zero_gradient (outlet): 대류만
            F = u_conv * face.area * nx_dir
            if F >= 0:
                system.add_diagonal(owner, F)

    return system


def _reference_solution(A, b):
    """직접법(spsolve)으로 참조 해를 계산."""
    from scipy.sparse.linalg import spsolve
    return spsolve(A, b)


def run_case16(results_dir="results", figures_dir="figures") -> dict:
    """
    전처리기 비교 검증 실행.

    Returns
    -------
    dict with keys: results, amg_available, best_preconditioner,
                    speedup_ilu0, speedup_amg, all_accurate
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 16: 전처리기 비교 검증")
    print("=" * 60)

    # 격자: 80x80 채널 - 고 Peclet 수(Pe=100)로 전처리기 효과가 뚜렷
    L, H = 1.0, 1.0
    nx, ny = 80, 80
    gamma_val = 0.01
    u_conv = 1.0
    Pe = u_conv * L / gamma_val
    print(f"  격자: {nx}x{ny} ({nx*ny} 셀), Pe={Pe:.0f}")
    mesh = generate_channel_mesh(L, H, nx, ny)

    # 선형 시스템 조립 (대류-확산)
    system = _build_convdiff_system(mesh, gamma_val=gamma_val, u_conv=u_conv)
    A = system.to_sparse()
    b = system.rhs.copy()
    n = system.n

    # 참조 해 (직접법)
    print("  참조 해(직접법) 계산 중...")
    T_ref = _reference_solution(A, b)
    denom_ref = max(float(np.sqrt(np.mean(T_ref**2))), 1e-15)

    preconditioners = ['none', 'jacobi', 'ilu0', 'iluk', 'amg']
    results = {}

    for pc_name in preconditioners:
        print(f"\n  전처리기: {pc_name}")

        # 전처리기 생성
        t_setup_start = time.time()
        try:
            M, pc_info = create_preconditioner(A, method=pc_name)
        except Exception as e:
            print(f"    전처리기 생성 실패: {e}")
            results[pc_name] = {
                'iterations': -1, 'wall_time': float('inf'),
                'L2_error': float('inf'), 'converged': False,
                'setup_time': 0.0
            }
            continue
        setup_time = time.time() - t_setup_start

        # 반복 횟수 카운터
        iter_count = [0]

        def callback(xk):
            iter_count[0] += 1

        # BiCGSTAB 풀기
        x0 = np.zeros(n)
        t_solve = time.time()
        try:
            x, info = scipy_bicgstab(
                A, b, x0=x0, atol=1e-10, rtol=0,
                maxiter=2000, M=M, callback=callback
            )
            converged = (info == 0)
        except Exception as e:
            print(f"    솔버 오류: {e}")
            results[pc_name] = {
                'iterations': -1, 'wall_time': float('inf'),
                'L2_error': float('inf'), 'converged': False,
                'setup_time': setup_time
            }
            continue
        wall_time = time.time() - t_solve + setup_time

        # L2 오차 (참조 해 대비)
        L2_error = float(np.sqrt(np.mean((x - T_ref)**2))) / denom_ref

        iters = iter_count[0]
        print(f"    반복: {iters}, 시간: {wall_time:.4f}s, L2 오차: {L2_error:.4e}, 수렴: {converged}")

        results[pc_name] = {
            'iterations': iters,
            'wall_time': wall_time,
            'L2_error': L2_error,
            'converged': converged,
            'setup_time': setup_time,
        }

    # 통계 계산
    all_accurate = all(
        r['L2_error'] < 0.01
        for r in results.values()
        if r['converged']
    )

    time_none = results['none']['wall_time']
    time_ilu0 = results['ilu0']['wall_time']
    time_amg = results.get('amg', {}).get('wall_time', float('inf'))

    speedup_ilu0 = time_none / time_ilu0 if time_ilu0 > 0 else 1.0
    speedup_amg = time_none / time_amg if time_amg > 0 and time_amg != float('inf') else 1.0

    # 최적 전처리기 (최소 wall_time, 수렴된 것)
    best = min(
        [(k, v) for k, v in results.items() if v['converged']],
        key=lambda kv: kv[1]['wall_time'],
        default=('none', results['none'])
    )
    best_preconditioner = best[0]

    print(f"\n  최적 전처리기: {best_preconditioner}")
    print(f"  ILU0 속도향상: {speedup_ilu0:.2f}x")
    print(f"  AMG 속도향상: {speedup_amg:.2f}x")
    print(f"  모두 정확: {all_accurate}")

    # 시각화: 반복 횟수 및 시간 바 차트
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pc_labels = [k for k in preconditioners if results[k]['iterations'] >= 0]
    iters_vals = [results[k]['iterations'] for k in pc_labels]
    time_vals = [results[k]['wall_time'] for k in pc_labels]

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    ax = axes[0]
    bars = ax.bar(pc_labels, iters_vals, color=colors[:len(pc_labels)])
    ax.set_xlabel('전처리기')
    ax.set_ylabel('반복 횟수')
    ax.set_title('BiCGSTAB 반복 횟수 비교')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, iters_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(pc_labels, time_vals, color=colors[:len(pc_labels)])
    ax.set_xlabel('전처리기')
    ax.set_ylabel('총 시간 (s)')
    ax.set_title('총 계산 시간 비교 (설정+풀기)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, time_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Case 16: 전처리기 비교 ({nx}x{ny} Conv-Diff Pe={Pe:.0f})', fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case16_preconditioner.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # VTU export (참조 해)
    case_dir = os.path.join(results_dir, 'case16')
    os.makedirs(case_dir, exist_ok=True)
    try:
        cell_data = {'T_reference': T_ref}
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case16 export skipped: {e}")

    return {
        'results': results,
        'amg_available': HAS_PYAMG,
        'best_preconditioner': best_preconditioner,
        'speedup_ilu0': float(speedup_ilu0),
        'speedup_amg': float(speedup_amg),
        'all_accurate': bool(all_accurate),
    }


if __name__ == "__main__":
    r = run_case16()
    print(f"\n결과 요약:")
    for pc, data in r['results'].items():
        print(f"  {pc:10s}: iters={data['iterations']:4d}, "
              f"time={data['wall_time']:.4f}s, L2={data['L2_error']:.4e}")
    if r['all_accurate']:
        print("  통과: all_accurate=True")
    else:
        print("  실패: 일부 전처리기 오차 초과")
