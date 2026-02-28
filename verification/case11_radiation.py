"""
Case 11: P1 복사 1D 슬래브 검증.

광학적으로 두꺼운 매질에서 P1 근사 정확도 확인.
두 벽 사이 1D 슬래브, 좌측 T_hot, 우측 T_cold.
해석해: 광학적으로 두꺼운 극한에서 G ~ 4*sigma*T^4 (확산 근사).
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
from core.fields import ScalarField
from models.radiation import P1RadiationModel, SIGMA_SB
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json


def run_case11(results_dir: str = "results",
               figures_dir: str = "figures") -> dict:
    """
    P1 복사 1D 슬래브 검증.

    Returns
    -------
    result : {'L2_error': float, 'converged': bool, ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Case 11: P1 Radiation 1D Slab")
    print("=" * 60)

    # 파라미터
    Lx = 1.0
    Ly = 0.02
    nx = 50
    ny = 1
    kappa = 10.0    # 흡수 계수 (광학적으로 두꺼움: tau = kappa*L = 10)

    T_hot = 1000.0   # 좌측 벽 [K]
    T_cold = 300.0   # 우측 벽 [K]

    print(f"  광학 두께: tau = {kappa * Lx:.1f}")

    # 격자
    mesh = generate_channel_mesh(Lx, Ly, nx, ny)
    print(f"  {mesh.summary()}")

    # 온도장 (선형 분포 초기값)
    T = ScalarField(mesh, "T")
    x_cells = np.array([mesh.cells[ci].center[0] for ci in range(mesh.n_cells)])
    T.values = T_hot + (T_cold - T_hot) * x_cells / Lx

    # P1 모델
    p1 = P1RadiationModel(mesh, kappa=kappa)

    # 경계조건: Marshak BC
    p1.set_bc('inlet', 'marshak', T_wall=T_hot)
    p1.set_bc('outlet', 'marshak', T_wall=T_cold)
    p1.set_bc('wall_bottom', 'zero_gradient')
    p1.set_bc('wall_top', 'zero_gradient')

    # 초기 G 추정
    p1.G.values = 4.0 * SIGMA_SB * T.values ** 4

    # 풀기
    print("  해석 중...")
    result = p1.solve(T, max_iter=200, tol=1e-8)
    print(f"  수렴: {result['converged']}, 반복: {result['iterations']}")

    # 해석해: 광학적으로 두꺼운 극한 → G ≈ 4σT⁴
    # 정확한 P1 1D 해석해:
    # -d/dx(1/(3κ) dG/dx) + κG = 4κσT⁴
    # 온도가 선형이면: G_exact(x)를 ODE로 풀 수 있음
    # 간단 검증: 복사 소스가 합리적 범위인지 확인
    G_equilibrium = 4.0 * SIGMA_SB * T.values ** 4
    G_num = p1.G.values

    # 상대 편차
    L2_error = np.sqrt(np.mean((G_num - G_equilibrium) ** 2)) / max(
        np.max(np.abs(G_equilibrium)), 1e-15)

    # 복사 소스
    q_r = p1.compute_radiative_source(T)
    print(f"  G_num 범위: [{np.min(G_num):.2e}, {np.max(G_num):.2e}]")
    print(f"  G_eq  범위: [{np.min(G_equilibrium):.2e}, {np.max(G_equilibrium):.2e}]")
    print(f"  복사 소스 범위: [{np.min(q_r):.2e}, {np.max(q_r):.2e}]")
    print(f"  L2(G vs 4sigmaT4): {L2_error:.4e}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x_cells, G_num / 1e6, 'b-', linewidth=2, label='G (P1 numerical)')
    ax.plot(x_cells, G_equilibrium / 1e6, 'r--', linewidth=2,
            label=r'$4\sigma T^4$ (equilibrium)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel(r'G [MW/m$^2$]')
    ax.set_title(f'P1 Radiation (kappa={kappa}, tau={kappa*Lx:.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_cells, q_r / 1e3, 'g-', linewidth=2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel(r'$q_r$ [kW/m$^3$]')
    ax.set_title('Radiative Source Term')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'case11_radiation.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  그래프 저장: {fig_path}")

    # --- VTU / JSON export ---
    case_dir = os.path.join(results_dir, 'case11')
    os.makedirs(case_dir, exist_ok=True)
    sigma = float(SIGMA_SB)
    params = {'L': Lx, 'nx': nx, 'kappa': kappa,
              'T_hot': T_hot, 'T_cold': T_cold, 'sigma': sigma}
    export_input_json(params, os.path.join(case_dir, 'input.json'))
    try:
        cell_data = {
            'G_radiation': p1.G.values,
            'temperature': T.values,
        }
        export_mesh_to_vtu(mesh, os.path.join(case_dir, 'mesh.vtu'), cell_data)
    except Exception as e:
        print(f"  [VTU] case11 export skipped: {e}")

    return {
        'L2_error': L2_error,
        'converged': result['converged'],
        'iterations': result['iterations'],
        'optical_thickness': kappa * Lx,
        'figure_path': fig_path,
        'residuals': result['residuals'],
    }


if __name__ == "__main__":
    result = run_case11()
    print(f"\n  결과: L2(G vs equilibrium) = {result['L2_error']:.4e}")
    if result['converged']:
        print("  V P1 솔버 수렴 확인")
    else:
        print("  X P1 솔버 미수렴")
