"""
Case 6: MUSCL/TVD 스킴 검증.

1D 스칼라 이류 문제에서 1차 Upwind와 MUSCL (van_leer 제한자) 비교.
초기 조건: 계단 함수 (x<0.3이면 phi=1, 아니면 phi=0)
이류 방정식: dphi/dt + u*dphi/dx = 0  (u=1, 순수 대류)

시간 T=0.4 후 정확해: step shifted to x<0.7 (= 0.3 + u*T)
MUSCL이 Upwind보다 인터페이스를 더 날카롭게 재현하는지 확인.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.plot_config import _FONT_NAME  # 한글 폰트 설정
import os
import sys

from mesh.mesh_generator import generate_channel_mesh
from core.fields import ScalarField, VectorField
from core.fvm_operators import FVMSystem, convection_operator_upwind, temporal_operator
from core.interpolation import muscl_deferred_correction, compute_mass_flux
from core.gradient import green_gauss_gradient
from core.linear_solver import solve_linear_system
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def _compute_diffusion_width(x_centers: np.ndarray, phi: np.ndarray,
                              threshold_lo: float = 0.1,
                              threshold_hi: float = 0.9) -> float:
    """
    10%-90% 기준으로 인터페이스 확산 폭 계산.

    phi가 0->1로 올라가는 영역에서 10%~90% 구간의 x 폭 반환.
    """
    mask = (phi >= threshold_lo) & (phi <= threshold_hi)
    if not np.any(mask):
        # fallback: all above or below threshold — width = 0 (perfectly sharp)
        return 0.0
    x_in_zone = x_centers[mask]
    return float(x_in_zone[-1] - x_in_zone[0]) if len(x_in_zone) > 1 else 0.0


def _apply_transient_bcs(phi_field: ScalarField, mass_flux: np.ndarray,
                          mesh, system: FVMSystem):
    """
    순수 대류 경계조건 적용 (확산 없음):
    - inlet (유입 F<0 or upwind from boundary): phi=1 고정
    - outlet, wall: zero_gradient (대류 아웃플로우)

    convection_operator_upwind에서 이미 경계 유입 처리를 위해
    F<0인 경계면 기여를 누락하므로, 여기서 RHS에 직접 추가한다.
    """
    for bname, fids in mesh.boundary_patches.items():
        for local_idx, fid in enumerate(fids):
            face = mesh.faces[fid]
            owner = face.owner
            F = mass_flux[fid]
            if bname == 'inlet':
                phi_b = phi_field.boundary_values[bname][local_idx]
                if F < 0:
                    # 유입: RHS에 -F * phi_b (F 음수이므로 양의 기여)
                    system.add_source(owner, -F * phi_b)
            # outlet / wall: zero_gradient — F>0이면 upwind operator가 이미 처리


def _solve_upwind_step(mesh, phi_old: np.ndarray, mass_flux: np.ndarray,
                        phi_inlet: float, dt: float, rho: float) -> np.ndarray:
    """단일 시간 스텝: Backward Euler + Upwind 대류."""
    n = mesh.n_cells
    system = FVMSystem(n)

    # 시간항
    temporal_operator(mesh, rho, dt, phi_old, system)

    # 대류항 (Upwind)
    convection_operator_upwind(mesh, mass_flux, system)

    # 경계 조건 (inlet Dirichlet, 유입 flux)
    phi_field = ScalarField(mesh, name='phi', default=0.0)
    phi_field.values[:] = phi_old
    phi_field.boundary_values['inlet'][:] = phi_inlet
    _apply_transient_bcs(phi_field, mass_flux, mesh, system)

    return solve_linear_system(system, method='direct')


def _solve_muscl_step(mesh, phi_old: np.ndarray, mass_flux: np.ndarray,
                       phi_inlet: float, dt: float, rho: float,
                       n_inner: int = 3) -> np.ndarray:
    """
    단일 시간 스텝: Backward Euler + MUSCL (지연 보정).

    inner 루프로 지연 보정 수렴.
    """
    n = mesh.n_cells
    phi_cur = phi_old.copy()

    phi_field = ScalarField(mesh, name='phi', default=0.0)
    phi_field.boundary_values['inlet'][:] = phi_inlet

    for _ in range(n_inner):
        phi_field.values[:] = phi_cur

        grad_phi = green_gauss_gradient(phi_field)

        system = FVMSystem(n)
        temporal_operator(mesh, rho, dt, phi_old, system)
        convection_operator_upwind(mesh, mass_flux, system)
        _apply_transient_bcs(phi_field, mass_flux, mesh, system)

        # MUSCL 지연 보정 (내부면만)
        correction = muscl_deferred_correction(
            mesh, phi_field, mass_flux, grad_phi, limiter_name='van_leer'
        )
        system.rhs += correction

        phi_new = solve_linear_system(system, method='direct')
        phi_new = np.clip(phi_new, 0.0, 1.0)
        phi_cur = phi_new

    return phi_cur


def run_case6(results_dir: str = "results", figures_dir: str = "figures") -> dict:
    """
    MUSCL/TVD 검증 실행.

    Returns
    -------
    result : dict with keys:
        'upwind_diffusion_width', 'muscl_diffusion_width',
        'muscl_sharper', 'figure_path'
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 6: MUSCL/TVD 스킴 검증 (Upwind vs MUSCL van_leer)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 격자 생성: 100 x 1 채널 (Lx=1.0, Ly=0.02)
    # ------------------------------------------------------------------
    Lx = 1.0
    Ly = 0.02
    nx, ny = 100, 1
    rho = 1.0
    u_conv = 1.0        # 이류 속도
    T_final = 0.4       # 최종 시간 (step이 x=0.7까지 이동)
    phi_inlet = 1.0     # 입구 고정값

    print(f"  격자: {nx} x {ny} 채널 (Lx={Lx}, Ly={Ly})")
    mesh = generate_channel_mesh(Lx, Ly, nx, ny)
    n_cells = mesh.n_cells
    print(f"  셀 수: {n_cells}, 면 수: {mesh.n_faces}")

    # ------------------------------------------------------------------
    # 2. 속도장 및 질량유속
    # ------------------------------------------------------------------
    U = VectorField(mesh, name="U")
    U.values[:, 0] = u_conv
    U.values[:, 1] = 0.0
    for bname in mesh.boundary_patches:
        n_b = len(mesh.boundary_patches[bname])
        U.boundary_values[bname] = np.zeros((n_b, 2))
        U.boundary_values[bname][:, 0] = u_conv

    mass_flux = compute_mass_flux(U, rho, mesh)

    # ------------------------------------------------------------------
    # 3. 초기 조건: 계단 함수
    # ------------------------------------------------------------------
    x_centers = np.array([mesh.cells[ci].center[0] for ci in range(n_cells)])
    phi_init = np.where(x_centers < 0.3, 1.0, 0.0).astype(np.float64)

    # CFL 기반 dt 설정 (CFL ~= 0.5 for stability)
    dx = Lx / nx
    CFL = 0.4
    dt = CFL * dx / u_conv
    n_steps = int(np.ceil(T_final / dt))
    dt = T_final / n_steps
    print(f"  dt={dt:.5f}, n_steps={n_steps}, CFL={u_conv*dt/dx:.3f}")

    # ------------------------------------------------------------------
    # 4. 시간 전진: Upwind
    # ------------------------------------------------------------------
    print("\n  [Upwind] 시간 전진 중...")
    phi_uw = phi_init.copy()
    for step in range(n_steps):
        phi_uw = _solve_upwind_step(mesh, phi_uw, mass_flux, phi_inlet, dt, rho)
        phi_uw = np.clip(phi_uw, 0.0, 1.0)
    print(f"    완료: phi min={phi_uw.min():.4f}, max={phi_uw.max():.4f}")

    # ------------------------------------------------------------------
    # 5. 시간 전진: MUSCL (van_leer)
    # ------------------------------------------------------------------
    print("  [MUSCL] 시간 전진 중...")
    phi_mu = phi_init.copy()
    for step in range(n_steps):
        phi_mu = _solve_muscl_step(mesh, phi_mu, mass_flux, phi_inlet, dt, rho,
                                    n_inner=2)
        phi_mu = np.clip(phi_mu, 0.0, 1.0)
    print(f"    완료: phi min={phi_mu.min():.4f}, max={phi_mu.max():.4f}")

    # ------------------------------------------------------------------
    # 6. 정확해: 계단이 u*T 만큼 이동
    # ------------------------------------------------------------------
    x_step_exact = 0.3 + u_conv * T_final  # = 0.7
    x_exact = np.linspace(0.0, Lx, 500)
    phi_exact = np.where(x_exact < x_step_exact, 1.0, 0.0)

    # ------------------------------------------------------------------
    # 7. 확산 폭 계산 (인터페이스 10%-90%)
    # ------------------------------------------------------------------
    upwind_width = _compute_diffusion_width(x_centers, phi_uw)
    muscl_width = _compute_diffusion_width(x_centers, phi_mu)

    print(f"\n  정확해 인터페이스 위치: x = {x_step_exact:.3f}")
    print(f"  Upwind 확산 폭 (10%-90%): {upwind_width:.4f} m")
    print(f"  MUSCL  확산 폭 (10%-90%): {muscl_width:.4f} m")
    muscl_sharper = muscl_width <= upwind_width
    print(f"  MUSCL가 Upwind보다 날카롭거나 같음: {muscl_sharper}")

    # ------------------------------------------------------------------
    # 8. 결과 시각화
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 전체 프로파일
    ax = axes[0]
    ax.step(x_exact, phi_exact, 'k--', linewidth=1.5, label='Exact', where='post')
    ax.plot(x_centers, phi_uw, 'b-o', markersize=3, linewidth=1.2,
            label=f'Upwind (diffusion width={upwind_width:.3f} m)')
    ax.plot(x_centers, phi_mu, 'r-s', markersize=3, linewidth=1.2,
            label=f'MUSCL van_leer (width={muscl_width:.3f} m)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('phi [-]')
    ax.set_title(f'1D Scalar Advection at T={T_final} (Upwind vs MUSCL)')
    ax.legend(fontsize=8)
    ax.set_xlim([0.0, Lx])
    ax.set_ylim([-0.15, 1.3])
    ax.grid(True, alpha=0.3)

    # 인터페이스 근방 확대
    ax = axes[1]
    zoom_lo = max(0.0, x_step_exact - 0.25)
    zoom_hi = min(Lx, x_step_exact + 0.15)
    mask_zoom = (x_centers >= zoom_lo) & (x_centers <= zoom_hi)
    x_zoom = x_centers[mask_zoom]
    mask_exact_zoom = (x_exact >= zoom_lo) & (x_exact <= zoom_hi)

    ax.step(x_exact[mask_exact_zoom], phi_exact[mask_exact_zoom],
            'k--', linewidth=1.5, label='Exact', where='post')
    ax.plot(x_zoom, phi_uw[mask_zoom], 'b-o', markersize=5,
            linewidth=1.5, label='Upwind')
    ax.plot(x_zoom, phi_mu[mask_zoom], 'r-s', markersize=5,
            linewidth=1.5, label='MUSCL van_leer')
    ax.axvline(x_step_exact, color='gray', linestyle=':', linewidth=1,
               label=f'Exact interface (x={x_step_exact:.2f})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('phi [-]')
    ax.set_title('Interface Region (Zoomed)')
    ax.legend(fontsize=9)
    ax.set_xlim([zoom_lo, zoom_hi])
    ax.set_ylim([-0.1, 1.2])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case6_muscl.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # ------------------------------------------------------------------
    # 9. 결과 저장
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(results_dir, 'case6_muscl.npz'),
        x=x_centers,
        phi_upwind=phi_uw,
        phi_muscl=phi_mu,
        phi_exact=phi_exact,
        x_exact=x_exact,
    )

    result = {
        'upwind_diffusion_width': float(upwind_width),
        'muscl_diffusion_width': float(muscl_width),
        'muscl_sharper': bool(muscl_sharper),
        'figure_path': fig_path,
    }

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case6')
    os.makedirs(case_dir, exist_ok=True)
    params = {'Lx': Lx, 'Ly': Ly, 'nx': nx, 'ny': ny,
              'u_adv': u_conv, 'CFL': CFL, 'n_steps': n_steps}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {'phi_upwind': phi_uw, 'phi_muscl': phi_mu}
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case6 export skipped: {e}")

    return result


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = run_case6()
    print("\n  결과 요약:")
    print(f"    Upwind 확산 폭:  {result['upwind_diffusion_width']:.4f} m")
    print(f"    MUSCL  확산 폭:  {result['muscl_diffusion_width']:.4f} m")
    if result['muscl_sharper']:
        print("  MUSCL이 Upwind보다 날카로운 인터페이스를 보임 (검증 통과)")
    else:
        print("  주의: MUSCL 확산 폭이 Upwind보다 넓음 - 확인 필요")
