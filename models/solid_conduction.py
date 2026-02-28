"""
고체 영역 열전도 솔버.

ρ_s·c_s·∂T_s/∂t = ∇·(k_s·∇T_s) + q'''
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField
from core.fvm_operators import (FVMSystem, diffusion_operator,
                                 temporal_operator, source_term,
                                 apply_boundary_conditions, under_relax)
from core.linear_solver import solve_linear_system


class SolidConductionSolver:
    """
    고체 영역 열전도 솔버.

    정상/비정상 열전도 방정식을 FVM으로 이산화하여 풀다.
    """

    def __init__(self, mesh: FVMesh, cell_ids: list = None):
        """
        Parameters
        ----------
        mesh : 전체 메쉬 (고체+유체)
        cell_ids : 고체 영역 셀 인덱스 (None이면 전체)
        """
        self.mesh = mesh
        self.cell_ids = cell_ids if cell_ids else list(range(mesh.n_cells))

        # 고체 물성치 (구리 기본값)
        self.rho = 8960.0      # 밀도 [kg/m³]
        self.cp = 385.0        # 비열 [J/(kg·K)]
        self.k_s = 401.0       # 열전도도 [W/(m·K)]

        # 온도장 (전체 메쉬 기준)
        self.T = ScalarField(mesh, "temperature_solid")
        self.T.set_uniform(300.0)

        # 체적 발열원 [W/m³]
        self.q_vol = np.zeros(mesh.n_cells)

        # 경계조건
        self.bc_T: dict = {}

        # 솔버 설정
        self.alpha_T = 0.9  # 완화계수
        self.dt = 0.01

    def set_material(self, rho: float, cp: float, k: float):
        """물성치 설정."""
        self.rho = rho
        self.cp = cp
        self.k_s = k

    def set_heat_source(self, q: float, cell_ids: list = None):
        """체적 발열원 설정."""
        if cell_ids is None:
            cell_ids = self.cell_ids
        for ci in cell_ids:
            self.q_vol[ci] = q

    def solve_steady(self, max_iter: int = 200, tol: float = 1e-6) -> dict:
        """
        정상 열전도 해석.

        Returns
        -------
        {'converged': bool, 'iterations': int, 'T_max': float, 'T_min': float}
        """
        mesh = self.mesh
        n = mesh.n_cells

        for iteration in range(max_iter):
            system = FVMSystem(n)

            # 확산 계수
            gamma = ScalarField(mesh, "k_solid")
            gamma.set_uniform(self.k_s)

            # 확산항
            diffusion_operator(mesh, gamma, system)

            # 소스항
            source_term(mesh, self.q_vol, system)

            # 경계조건
            mass_flux = np.zeros(mesh.n_faces)
            apply_boundary_conditions(mesh, self.T, gamma, mass_flux,
                                     system, self.bc_T)

            # 완화
            under_relax(system, self.T, self.alpha_T)

            # 풀기
            T_old = self.T.values.copy()
            self.T.values = solve_linear_system(system, self.T.values,
                                                 method='direct')

            # 수렴 판정
            change = np.linalg.norm(self.T.values - T_old)
            norm = max(np.linalg.norm(self.T.values), 1e-15)
            res = change / norm

            if res < tol:
                return {
                    'converged': True,
                    'iterations': iteration + 1,
                    'T_max': float(self.T.values[self.cell_ids].max()),
                    'T_min': float(self.T.values[self.cell_ids].min())
                }

        return {
            'converged': False,
            'iterations': max_iter,
            'T_max': float(self.T.values[self.cell_ids].max()),
            'T_min': float(self.T.values[self.cell_ids].min())
        }

    def solve_one_step(self, dt: float = None):
        """비정상 1 시간 스텝 풀기."""
        if dt is not None:
            self.dt = dt

        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        self.T.store_old()

        # 확산
        gamma = ScalarField(mesh, "k_solid")
        gamma.set_uniform(self.k_s)
        diffusion_operator(mesh, gamma, system)

        # 시간항
        temporal_operator(mesh, self.rho * self.cp, self.dt,
                         self.T.old_values, system)

        # 소스항
        source_term(mesh, self.q_vol, system)

        # 경계조건
        mass_flux = np.zeros(mesh.n_faces)
        apply_boundary_conditions(mesh, self.T, gamma, mass_flux,
                                 system, self.bc_T)

        # 풀기
        self.T.values = solve_linear_system(system, self.T.values,
                                             method='direct')
