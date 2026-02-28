"""
1차 반응 모델 및 종(species) 수송 솔버.

A → B, 반응률: R_A = -k_r * C_A * rho
플러그 흐름 반응기 해석해: C_A(x) = C_A0 * exp(-k_r * x / u)
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField, VectorField
from core.fvm_operators import (FVMSystem, diffusion_operator,
                                 convection_operator_upwind,
                                 linearized_source, apply_boundary_conditions,
                                 under_relax)
from core.linear_solver import solve_linear_system


class FirstOrderReaction:
    """
    1차 비가역 반응 A → B.

    Parameters
    ----------
    k_r : 반응 속도 상수 [1/s]
    """

    def __init__(self, k_r: float = 1.0):
        self.k_r = k_r

    def reaction_rate(self, C_A: np.ndarray, rho: float) -> np.ndarray:
        """
        종 A 소멸률: R_A = -k_r * C_A (체적당 소스)

        Parameters
        ----------
        C_A : (n,) 종 A 질량분율
        rho : 밀도

        Returns
        -------
        R_A : (n,) 반응 소스 [1/s * 질량분율]
        """
        return -self.k_r * C_A

    def source_linearization(self, C_A: np.ndarray, rho: float):
        """
        선형화: R = Su + Sp * C_A
        Su = 0, Sp = -k_r (안정적: 음수)

        Returns
        -------
        Su, Sp : (n,) arrays
        """
        n = len(C_A)
        Su = np.zeros(n)
        Sp = np.full(n, -self.k_r)
        return Su, Sp


class SpeciesTransportSolver:
    """
    종 수송 방정식 솔버.

    ∂(ρ·C)/∂t + ∇·(ρ·u·C) = ∇·(ρ·D·∇C) + R

    Parameters
    ----------
    mesh : FVMesh
    rho : 밀도
    D : 확산 계수 [m²/s]
    reaction : 반응 모델 (None이면 반응 없음)
    """

    def __init__(self, mesh: FVMesh, rho: float = 1.0,
                 D: float = 1e-5, reaction=None):
        self.mesh = mesh
        self.rho = rho
        self.D = D
        self.reaction = reaction

        self.C = ScalarField(mesh, "C_A")
        self.bc_C: dict = {}
        self.alpha_C = 0.8  # 완화계수

    def set_bc(self, patch_name: str, bc_type: str, value=None):
        """경계조건 설정."""
        if bc_type == 'dirichlet':
            self.bc_C[patch_name] = {'type': 'dirichlet'}
            if value is not None:
                fids = self.mesh.boundary_patches.get(patch_name, [])
                self.C.boundary_values[patch_name] = np.full(len(fids), value)
        elif bc_type == 'zero_gradient':
            self.bc_C[patch_name] = {'type': 'zero_gradient'}

    def solve_steady(self, U: VectorField, mass_flux: np.ndarray,
                     max_iter: int = 200, tol: float = 1e-6) -> dict:
        """
        정상 상태 종 수송 풀이.

        Parameters
        ----------
        U : 속도장
        mass_flux : 면 질량유속
        max_iter : 최대 반복
        tol : 수렴 허용오차

        Returns
        -------
        result : {'converged': bool, 'iterations': int, 'residuals': list}
        """
        mesh = self.mesh
        n = mesh.n_cells
        residuals = []

        # 확산 계수 필드
        gamma = ScalarField(mesh, "D_eff")
        gamma.values[:] = self.rho * self.D

        for it in range(max_iter):
            system = FVMSystem(n)

            # 확산
            diffusion_operator(mesh, gamma, system)

            # 대류 (Upwind)
            convection_operator_upwind(mesh, mass_flux, system)

            # 반응 소스
            if self.reaction is not None:
                Su, Sp = self.reaction.source_linearization(
                    self.C.values, self.rho)
                linearized_source(mesh, Sp, Su, system)

            # 경계조건
            apply_boundary_conditions(mesh, self.C, gamma, mass_flux,
                                       system, self.bc_C)

            # 완화
            under_relax(system, self.C, self.alpha_C)

            # 풀기
            C_new = solve_linear_system(system, self.C.values, method='direct')

            # 잔차
            res = np.sqrt(np.mean((C_new - self.C.values) ** 2))
            residuals.append(res)
            self.C.values = np.maximum(C_new, 0.0)

            if it > 0 and res < tol:
                return {'converged': True, 'iterations': it + 1,
                        'residuals': residuals}

        return {'converged': False, 'iterations': max_iter,
                'residuals': residuals}
