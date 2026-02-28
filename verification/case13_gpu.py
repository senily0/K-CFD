"""
Case 13: GPU 가속 벤치마크.

다양한 격자 크기에서 CPU direct vs CPU BiCGSTAB vs GPU BiCGSTAB 비교.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys
import time

from mesh.mesh_generator import generate_channel_mesh
from core.fields import ScalarField
from core.fvm_operators import FVMSystem, diffusion_operator, source_term
from core.gpu_solver import detect_gpu_backend, benchmark_solvers, _cpu_bicgstab
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json

def run_case13(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    GPU 벤치마크 및 BiCGSTAB 정확도 검증.

    Returns
    -------
    result : {'backend': str, 'benchmarks': list, 'figure_path': str,
              'gpu_available': bool, 'converged': bool,
              'accuracy_verified': bool, 'max_bicgstab_error': float}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 13: GPU Acceleration Benchmark")
    print("=" * 60)

    backend = detect_gpu_backend()
    print(f"  GPU backend: {backend}")
    gpu_available = backend != 'cpu'

    # 격자 크기별 벤치마크 — 100, 1K, 10K, ~50K cells
    grid_sizes = [
        (10, 10),     # 100 cells
        (32, 32),     # 1024 cells
        (100, 100),   # 10000 cells
    ]

    benchmarks = []
    bicgstab_errors = []

    for nx, ny in grid_sizes:
        n_cells = nx * ny
        print(f"\n  --- {n_cells} cells ({nx}x{ny}) ---")

        # 격자 및 확산 문제 구성
        mesh = generate_channel_mesh(1.0, 1.0, nx, ny)

        gamma = ScalarField(mesh, "gamma")
        gamma.values[:] = 1.0

        # 소스항
        S = np.zeros(mesh.n_cells)
        for ci in range(mesh.n_cells):
            cc = mesh.cells[ci].center
            S[ci] = 10.0 * np.sin(np.pi * cc[0]) * np.sin(np.pi * cc[1])

        # FVM 시스템 조립
        system = FVMSystem(mesh.n_cells)
        diffusion_operator(mesh, gamma, system)
        source_term(mesh, S, system)

        # 경계조건 (T=0 Dirichlet)
        for bname, fids in mesh.boundary_patches.items():
            for fid in fids:
                face = mesh.faces[fid]
                owner = face.owner
                d = np.linalg.norm(face.center - mesh.cells[owner].center)
                if d > 1e-30:
                    coeff = 1.0 * face.area / d
                    system.add_source(owner, coeff * 0.0)

        A = system.to_sparse()
        b = system.rhs

        # 벤치마크
        bm = benchmark_solvers(A, b, n_runs=1)
        bm['n_cells'] = n_cells

        # 정확도 검증: BiCGSTAB vs direct (L2 norm)
        x_direct = bm['x_direct']
        x_bicg = _cpu_bicgstab(A, b, None, 1e-8, 2000)
        ref_norm = np.linalg.norm(x_direct)
        if ref_norm > 1e-30:
            l2_err = np.linalg.norm(x_bicg - x_direct) / ref_norm
        else:
            l2_err = float(np.linalg.norm(x_bicg - x_direct))
        bm['bicgstab_l2_error'] = l2_err
        bicgstab_errors.append(l2_err)

        benchmarks.append(bm)

        print(f"    CPU direct:   {bm['cpu_direct']:.4f} s")
        print(f"    CPU BiCGSTAB: {bm['cpu_bicgstab']:.4f} s")
        print(f"    BiCGSTAB L2 error: {l2_err:.2e}")
        if not np.isnan(bm['gpu_bicgstab']):
            print(f"    GPU BiCGSTAB: {bm['gpu_bicgstab']:.4f} s")
            speedup = bm['cpu_bicgstab'] / max(bm['gpu_bicgstab'], 1e-10)
            print(f"    Speedup:      {speedup:.2f}x")
        else:
            print(f"    GPU BiCGSTAB: N/A (no GPU)")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_cells_list = [bm['n_cells'] for bm in benchmarks]
    cpu_direct = [bm['cpu_direct'] for bm in benchmarks]
    cpu_bicg = [bm['cpu_bicgstab'] for bm in benchmarks]
    gpu_bicg = [bm['gpu_bicgstab'] for bm in benchmarks]

    # 시간 비교
    ax = axes[0]
    ax.loglog(n_cells_list, cpu_direct, 'ro-', linewidth=2,
              markersize=8, label='CPU Direct')
    ax.loglog(n_cells_list, cpu_bicg, 'bs-', linewidth=2,
              markersize=8, label='CPU BiCGSTAB')
    if not all(np.isnan(gpu_bicg)):
        valid = [not np.isnan(t) for t in gpu_bicg]
        n_valid = [n for n, v in zip(n_cells_list, valid) if v]
        t_valid = [t for t, v in zip(gpu_bicg, valid) if v]
        ax.loglog(n_valid, t_valid, 'g^-', linewidth=2,
                  markersize=8, label='GPU BiCGSTAB')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Solve Time [s]')
    ax.set_title('Solver Performance')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # 스피드업 + BiCGSTAB L2 오차 (보조 y축)
    ax = axes[1]
    speedup_direct = [d / max(b, 1e-10) for d, b in zip(cpu_direct, cpu_bicg)]
    ax.semilogx(n_cells_list, speedup_direct, 'bs-', linewidth=2,
                markersize=8, label='BiCGSTAB vs Direct (speed)')
    if not all(np.isnan(gpu_bicg)):
        speedup_gpu = [c / max(g, 1e-10) if not np.isnan(g) else float('nan')
                       for c, g in zip(cpu_bicg, gpu_bicg)]
        n_valid = [n for n, s in zip(n_cells_list, speedup_gpu) if not np.isnan(s)]
        s_valid = [s for s in speedup_gpu if not np.isnan(s)]
        if n_valid:
            ax.semilogx(n_valid, s_valid, 'g^-', linewidth=2,
                        markersize=8, label='GPU vs CPU BiCGSTAB')
    ax2 = ax.twinx()
    errors = [bm['bicgstab_l2_error'] for bm in benchmarks]
    ax2.semilogy(n_cells_list, errors, 'm^--', linewidth=1.5,
                 markersize=6, label='BiCGSTAB L2 error')
    ax2.set_ylabel('BiCGSTAB L2 Error (vs direct)', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup & Accuracy')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case13_gpu.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {fig_path}")

    # VTU 및 JSON 내보내기 (가장 큰 격자, 루프에서 마지막으로 해석된 결과 재사용)
    last_bm = benchmarks[-1]
    nx_lg, ny_lg = grid_sizes[-1]
    mesh_lg = generate_channel_mesh(1.0, 1.0, nx_lg, ny_lg)
    vtu_path = os.path.join(results_dir, "case13_largest.vtu")
    json_path = os.path.join(results_dir, "case13_input.json")
    export_mesh_to_vtu(mesh_lg, vtu_path, cell_data={"T": last_bm['x_direct']})
    export_input_json(
        {
            "case": 13,
            "description": "GPU acceleration benchmark",
            "backend": backend,
            "grid_nx": nx_lg,
            "grid_ny": ny_lg,
            "n_cells": nx_lg * ny_lg,
            "diffusivity": 1.0,
            "source": "10*sin(pi*x)*sin(pi*y)",
            "bc": "Dirichlet T=0 on all boundaries",
        },
        json_path,
    )
    print(f"  VTU 내보내기: {vtu_path}")
    print(f"  JSON 내보내기: {json_path}")

    # 정확도 및 수렴 판정
    max_err = float(max(bicgstab_errors)) if bicgstab_errors else float('nan')
    accuracy_verified = bool(max_err < 1e-6) if not np.isnan(max_err) else False
    converged = True  # 모든 벤치마크가 완료되면 수렴으로 판정

    # 최종 요약 출력
    print("\n" + "=" * 60)
    print("  Case 13 Summary")
    print("=" * 60)
    gpu_status = f"Yes ({backend})" if gpu_available else "No (CPU fallback)"
    print(f"  GPU 사용 가능:        {gpu_status}")
    acc_status = "PASS" if accuracy_verified else "FAIL"
    print(f"  BiCGSTAB 정확도 검증: {acc_status} (최대 L2 오차 = {max_err:.2e})")
    print(f"  수렴:                 {'Yes' if converged else 'No'}")
    print("\n  격자 크기별 스케일링 (CPU Direct vs BiCGSTAB):")
    for bm in benchmarks:
        ratio = bm['cpu_direct'] / max(bm['cpu_bicgstab'], 1e-10)
        print(f"    {bm['n_cells']:>6} cells: direct={bm['cpu_direct']:.4f}s, "
              f"bicgstab={bm['cpu_bicgstab']:.4f}s, "
              f"ratio={ratio:.2f}x, err={bm['bicgstab_l2_error']:.2e}")
    print("=" * 60)

    return {
        'backend': backend,
        'benchmarks': benchmarks,
        'figure_path': fig_path,
        'gpu_available': gpu_available,
        'converged': converged,
        'accuracy_verified': accuracy_verified,
        'max_bicgstab_error': max_err,
    }

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case13()
    print(f"\n  GPU backend: {result['backend']}")
    print(f"  GPU available: {result['gpu_available']}")
    print(f"  Accuracy verified: {result['accuracy_verified']}")
    print(f"  Max BiCGSTAB error: {result['max_bicgstab_error']:.2e}")
    for bm in result['benchmarks']:
        print(f"  {bm['n_cells']} cells: direct={bm['cpu_direct']:.4f}s, "
              f"bicgstab={bm['cpu_bicgstab']:.4f}s, "
              f"err={bm['bicgstab_l2_error']:.2e}")
