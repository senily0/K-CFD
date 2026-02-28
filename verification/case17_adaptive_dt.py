"""
Case 17: CFL 기반 적응 시간 간격 검증.

고정 dt와 적응 dt를 단순 이류-확산 문제에서 비교.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정

from mesh.mesh_generator import generate_channel_mesh
from core.time_control import AdaptiveTimeControl

def run_case17(results_dir="results", figures_dir="figures"):
    """적응 시간 간격 검증."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'case17'), exist_ok=True)

    print("=" * 60)
    print("Case 17: 적응 시간 간격 (CFL) 검증")
    print("=" * 60)

    # 1. Create mesh
    mesh = generate_channel_mesh(length=1.0, height=0.1, nx=100, ny=5)
    n = mesh.n_cells
    ndim = getattr(mesh, 'ndim', 2)

    # 2. Set up velocity field (non-uniform: parabolic profile, max=1.0)
    velocity = np.zeros((n, ndim))
    for i in range(n):
        y = mesh.cells[i].center[1]
        velocity[i, 0] = 4.0 * y * (0.1 - y) / (0.1 ** 2)  # parabolic, max=1.0

    # 3. Fixed dt reference
    dt_fixed = 0.001
    n_steps_fixed = int(0.2 / dt_fixed)  # 200 steps

    # 4. Adaptive dt run (compute_dt loop only, no PDE solve)
    tc = AdaptiveTimeControl(dt_init=0.001, dt_min=1e-6, dt_max=0.005,
                             cfl_target=0.5, cfl_max=1.0)

    t = 0.0
    n_steps_adaptive = 0
    t_end = 0.2
    while t < t_end - 1e-15:
        new_dt, info = tc.compute_dt(mesh, velocity, alpha_diff=0.001)
        t += new_dt
        n_steps_adaptive += 1
        if n_steps_adaptive > 10000:  # safety
            break

    # 5. Verify CFL always below max
    cfl_always_ok = all(c <= 1.0 + 1e-10 for c in tc.cfl_history)

    # 6. Simple explicit upwind advection with fixed dt
    dx_arr = np.array([mesh.cells[i].volume ** 0.5 for i in range(n)])

    phi_init = np.zeros(n)
    for i in range(n):
        if mesh.cells[i].center[0] < 0.3:
            phi_init[i] = 1.0

    phi_fixed = phi_init.copy()
    for step in range(n_steps_fixed):
        phi_new = phi_fixed.copy()
        for i in range(1, n):
            u = velocity[i, 0]
            dx = dx_arr[i]
            phi_new[i] = phi_fixed[i] - u * dt_fixed / dx * (phi_fixed[i] - phi_fixed[i - 1])
        phi_fixed = phi_new

    # 7. Same advection with adaptive dt
    phi_adaptive = phi_init.copy()
    tc2 = AdaptiveTimeControl(dt_init=0.001, dt_min=1e-6, dt_max=0.005, cfl_target=0.5)
    t = 0.0
    adaptive_steps = 0
    while t < 0.2 - 1e-15:
        new_dt, _ = tc2.compute_dt(mesh, velocity, alpha_diff=0.001)
        phi_new = phi_adaptive.copy()
        for i in range(1, n):
            u = velocity[i, 0]
            dx = dx_arr[i]
            phi_new[i] = phi_adaptive[i] - u * new_dt / dx * (phi_adaptive[i] - phi_adaptive[i - 1])
        phi_adaptive = phi_new
        t += new_dt
        adaptive_steps += 1
        if adaptive_steps > 10000:
            break

    # 8. Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # dt history
    axes[0, 0].plot(tc.dt_history)
    axes[0, 0].set_title('dt History (Adaptive)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('dt [s]')
    axes[0, 0].grid(True)

    # CFL history
    axes[0, 1].plot(tc.cfl_history)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='CFL=1.0')
    axes[0, 1].set_title('CFL History')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('CFL')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Solutions comparison
    x_coords = [mesh.cells[i].center[0] for i in range(n)]
    axes[1, 0].plot(x_coords, phi_fixed, label='Fixed dt')
    axes[1, 0].plot(x_coords, phi_adaptive, '--', label='Adaptive dt')
    axes[1, 0].set_title('Solution Comparison (phi)')
    axes[1, 0].set_xlabel('x [m]')
    axes[1, 0].set_ylabel('phi')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Summary text
    axes[1, 1].axis('off')
    summary = (
        f"Fixed dt={dt_fixed}: {n_steps_fixed} steps\n"
        f"Adaptive dt: {adaptive_steps} steps\n"
        f"Step ratio: {adaptive_steps / n_steps_fixed:.2f}\n"
        f"CFL always <= 1.0: {cfl_always_ok}\n"
        f"dt range: [{min(tc.dt_history):.6f}, {max(tc.dt_history):.6f}]"
    )
    axes[1, 1].text(0.1, 0.5, summary, fontsize=13, family='monospace', va='center')

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case17_adaptive_dt.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # 9. Export VTU
    try:
        from mesh.vtk_exporter import export_mesh_to_vtu
        export_mesh_to_vtu(
            mesh,
            os.path.join(results_dir, 'case17', 'mesh.vtu'),
            cell_data={'phi_adaptive': phi_adaptive, 'velocity_x': velocity[:, 0]}
        )
        print("  VTU exported.")
    except Exception as e:
        print(f"  VTU export skipped: {e}")

    result = {
        'fixed_dt_steps': n_steps_fixed,
        'adaptive_dt_steps': adaptive_steps,
        'step_reduction': adaptive_steps / n_steps_fixed,
        'cfl_max_history': tc.cfl_history,
        'dt_history': tc.dt_history,
        'cfl_always_below_max': cfl_always_ok,
        'converged': True,
    }

    print(f"  Fixed dt steps : {n_steps_fixed}")
    print(f"  Adaptive steps : {adaptive_steps}")
    print(f"  CFL always OK  : {cfl_always_ok}")
    return result

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_case17()
