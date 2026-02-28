"""
P1 복사 근사 모델.

지배 방정식:
  -∇·(1/(3κ) · ∇G) + κ·G = 4·κ·σ·T⁴

여기서:
  G = 입사 복사 (incident radiation) [W/m²]
  κ = 흡수 계수 [1/m]
  σ = Stefan-Boltzmann 상수
  T = 온도 [K]

Marshak 경계조건:
  G + 2/(3κ) · ∂G/∂n = 4·σ·T_wall⁴
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField
from core.fvm_operators import (FVMSystem, diffusion_operator,
                                 linearized_source, apply_boundary_conditions)
from core.linear_solver import solve_linear_system


SIGMA_SB = 5.67e-8  # Stefan-Boltzmann 상수 [W/(m²·K⁴)]


class P1RadiationModel:
    """
    P1 복사 근사 모델.

    Parameters
    ----------
    mesh : FVMesh
    kappa : 흡수 계수 [1/m]
    """

    def __init__(self, mesh: FVMesh, kappa: float = 1.0):
        self.mesh = mesh
        self.kappa = kappa

        # 입사 복사 필드
        self.G = ScalarField(mesh, "G")
        self.bc_G: dict = {}

        # 완화계수
        self.alpha_G = 0.8

    def solve(self, T: ScalarField, max_iter: int = 100,
              tol: float = 1e-6) -> dict:
        """
        P1 방정식 풀이: -∇·(Γ∇G) + κG = 4κσT⁴

        Parameters
        ----------
        T : 온도장 [K]
        max_iter : 최대 반복
        tol : 수렴 허용오차

        Returns
        -------
        result : {'converged': bool, 'iterations': int, 'residuals': list}
        """
        mesh = self.mesh
        n = mesh.n_cells
        kappa = self.kappa
        residuals = []

        # 확산 계수: Γ = 1/(3κ)
        gamma = ScalarField(mesh, "gamma_rad")
        gamma.values[:] = 1.0 / (3.0 * kappa)

        # 소스항: 4κσT⁴ (Su), -κ (Sp, 선형화)
        # S = 4κσT⁴ - κG  →  Su = 4κσT⁴, Sp = -κ
        emission = 4.0 * kappa * SIGMA_SB * T.values ** 4

        for it in range(max_iter):
            system = FVMSystem(n)

            # 확산: -∇·(Γ∇G)
            diffusion_operator(mesh, gamma, system)

            # 소스: Su = 4κσT⁴, Sp = -κ
            Su = emission.copy()
            Sp = np.full(n, -kappa)
            linearized_source(mesh, Sp, Su, system)

            # 경계조건
            apply_boundary_conditions(mesh, self.G, gamma,
                                       np.zeros(mesh.n_faces),
                                       system, self.bc_G)

            # 풀기
            G_new = solve_linear_system(system, self.G.values, method='direct')

            # 잔차
            res = np.sqrt(np.mean((G_new - self.G.values) ** 2))
            residuals.append(res)

            self.G.values = G_new.copy()

            if res < tol:
                return {'converged': True, 'iterations': it + 1,
                        'residuals': residuals}

        return {'converged': False, 'iterations': max_iter,
                'residuals': residuals}

    def compute_radiative_source(self, T: ScalarField) -> np.ndarray:
        """
        복사 에너지 소스항: q_r = κ·(G - 4σT⁴)

        에너지 방정식 우변에 추가.

        Returns
        -------
        q_r : (n_cells,) [W/m³]
        """
        return self.kappa * (self.G.values - 4.0 * SIGMA_SB * T.values ** 4)

    def set_bc(self, patch_name: str, bc_type: str, T_wall: float = None):
        """
        G 경계조건 설정.

        Marshak BC: G ≈ 4σT_wall⁴ (단순화)
        """
        if bc_type == 'marshak' and T_wall is not None:
            self.bc_G[patch_name] = {'type': 'dirichlet'}
            fids = self.mesh.boundary_patches.get(patch_name, [])
            G_wall = 4.0 * SIGMA_SB * T_wall ** 4
            self.G.boundary_values[patch_name] = np.full(len(fids), G_wall)
        elif bc_type == 'zero_gradient':
            self.bc_G[patch_name] = {'type': 'zero_gradient'}
        elif bc_type == 'dirichlet':
            self.bc_G[patch_name] = {'type': 'dirichlet'}
