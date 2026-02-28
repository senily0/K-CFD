"""
표준 k-epsilon 난류 모델 + 벽함수.

상수: C_μ=0.09, C1ε=1.44, C2ε=1.92, σ_k=1.0, σ_ε=1.3
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField, VectorField
from core.fvm_operators import (FVMSystem, diffusion_operator,
                                 convection_operator_upwind, temporal_operator,
                                 linearized_source, apply_boundary_conditions,
                                 under_relax)
from core.gradient import green_gauss_gradient
from core.linear_solver import solve_linear_system


# 표준 k-epsilon 상수
C_MU = 0.09
C1_EPS = 1.44
C2_EPS = 1.92
SIGMA_K = 1.0
SIGMA_EPS = 1.3
KAPPA = 0.41   # von Karman 상수
E_WALL = 9.793  # 벽함수 상수


class KEpsilonModel:
    """
    표준 k-epsilon 난류 모델.

    ∂(ρk)/∂t + ∇·(ρuk) = ∇·((μ + μ_t/σ_k)∇k) + P_k - ρε
    ∂(ρε)/∂t + ∇·(ρuε) = ∇·((μ + μ_t/σ_ε)∇ε) + C1ε·(ε/k)·P_k - C2ε·ρε²/k
    μ_t = ρ·C_μ·k²/ε
    """

    def __init__(self, mesh: FVMesh, rho: float = 1.0, mu: float = 1e-3):
        self.mesh = mesh
        self.rho = rho
        self.mu = mu

        # 난류 필드
        self.k = ScalarField(mesh, "k")
        self.epsilon = ScalarField(mesh, "epsilon")
        self.mu_t = ScalarField(mesh, "mu_t")  # 난류 점성

        # 초기값
        k_init = 1e-4
        eps_init = 1e-5
        self.k.set_uniform(k_init)
        self.epsilon.set_uniform(eps_init)
        self._update_mu_t()

        # 완화계수
        self.alpha_k = 0.7
        self.alpha_eps = 0.7

        # 벽면 패치 이름들
        self.wall_patches: list = []

        # 경계조건
        self.bc_k: dict = {}
        self.bc_eps: dict = {}

    def initialize(self, k_init: float = 1e-4, eps_init: float = 1e-5,
                   intensity: float = None, length_scale: float = None,
                   U_ref: float = None):
        """
        난류 필드 초기화.

        intensity와 U_ref가 주어지면: k = 1.5*(I*U)^2, ε = C_μ^0.75 * k^1.5 / l
        """
        if intensity is not None and U_ref is not None:
            k_init = 1.5 * (intensity * U_ref) ** 2
            if length_scale is not None:
                eps_init = C_MU**0.75 * k_init**1.5 / length_scale
            else:
                eps_init = C_MU**0.75 * k_init**1.5 / 0.01

        self.k.set_uniform(max(k_init, 1e-10))
        self.epsilon.set_uniform(max(eps_init, 1e-10))
        self._update_mu_t()

    def set_wall_patches(self, patches: list):
        """벽면 패치 지정."""
        self.wall_patches = patches
        for p in patches:
            self.bc_k[p] = {'type': 'zero_gradient'}
            self.bc_eps[p] = {'type': 'zero_gradient'}

    def solve(self, U: VectorField, mass_flux: np.ndarray, dt: float = None):
        """
        k, ε 수송 방정식을 풀고 μ_t 갱신.

        Parameters
        ----------
        U : 속도장
        mass_flux : 면 질량유속
        dt : 시간 간격 (비정상 시)
        """
        # 생산항 P_k 계산
        P_k = self._compute_production(U)

        # k 방정식
        self._solve_k(mass_flux, P_k, dt)

        # ε 방정식
        self._solve_epsilon(mass_flux, P_k, dt)

        # μ_t 갱신
        self._update_mu_t()

        # 벽함수 적용
        self._apply_wall_functions(U)

    def _compute_production(self, U: VectorField) -> np.ndarray:
        """
        난류 생산항: P_k = μ_t · S² (S = strain rate magnitude)
        ndim 차원에 대한 일반화된 변형률 텐서.
        """
        mesh = self.mesh
        ndim = getattr(mesh, 'ndim', 2)
        n = mesh.n_cells

        # 속도 성분별 기울기 계산
        grads = []  # grads[i] = grad(U_i), shape (n, ndim)
        for i in range(ndim):
            comp_field = ScalarField(mesh, f"u{i}")
            comp_field.values = U.values[:, i].copy()
            for bname in U.boundary_values:
                comp_field.boundary_values[bname] = U.boundary_values[bname][:, i].copy()
            grads.append(green_gauss_gradient(comp_field))

        # S² = 2*(S_ij·S_ij), S_ij = 0.5*(dU_i/dx_j + dU_j/dx_i)
        S_sq = np.zeros(n)
        for i in range(ndim):
            for j in range(ndim):
                Sij = 0.5 * (grads[i][:, j] + grads[j][:, i])
                S_sq += 2.0 * Sij**2

        P_k = self.mu_t.values * S_sq
        return P_k

    def _solve_k(self, mass_flux: np.ndarray, P_k: np.ndarray, dt: float = None):
        """k 수송 방정식."""
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        # 확산 계수: μ + μ_t/σ_k
        gamma_k = ScalarField(mesh, "gamma_k")
        gamma_k.values = self.mu + self.mu_t.values / SIGMA_K

        # 확산
        diffusion_operator(mesh, gamma_k, system)

        # 대류
        convection_operator_upwind(mesh, mass_flux, system)

        # 시간항
        if dt is not None:
            self.k.store_old()
            temporal_operator(mesh, self.rho, dt, self.k.old_values, system)

        # 소스: P_k - ρε
        # 선형화: Su = P_k, Sp = -ρε/k (Sp는 음수 → 안정적)
        Su = P_k.copy()
        Sp = np.zeros(n)
        for ci in range(n):
            if self.k.values[ci] > 1e-15:
                Sp[ci] = -self.rho * self.epsilon.values[ci] / self.k.values[ci]
        linearized_source(mesh, Sp, Su, system)

        # 경계조건
        apply_boundary_conditions(mesh, self.k, gamma_k, mass_flux, system, self.bc_k)

        # 완화
        under_relax(system, self.k, self.alpha_k)

        # 풀기
        self.k.values = solve_linear_system(system, self.k.values, method='bicgstab')
        self.k.values = np.maximum(self.k.values, 1e-10)

    def _solve_epsilon(self, mass_flux: np.ndarray, P_k: np.ndarray, dt: float = None):
        """ε 수송 방정식."""
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        # 확산 계수
        gamma_eps = ScalarField(mesh, "gamma_eps")
        gamma_eps.values = self.mu + self.mu_t.values / SIGMA_EPS

        # 확산
        diffusion_operator(mesh, gamma_eps, system)

        # 대류
        convection_operator_upwind(mesh, mass_flux, system)

        # 시간항
        if dt is not None:
            self.epsilon.store_old()
            temporal_operator(mesh, self.rho, dt, self.epsilon.old_values, system)

        # 소스: C1ε·(ε/k)·P_k - C2ε·ρε²/k
        Su = np.zeros(n)
        Sp = np.zeros(n)
        for ci in range(n):
            k_val = max(self.k.values[ci], 1e-10)
            eps_val = max(self.epsilon.values[ci], 1e-10)
            Su[ci] = C1_EPS * (eps_val / k_val) * P_k[ci]
            Sp[ci] = -C2_EPS * self.rho * eps_val / k_val
        linearized_source(mesh, Sp, Su, system)

        # 경계조건
        apply_boundary_conditions(mesh, self.epsilon, gamma_eps, mass_flux, system, self.bc_eps)

        # 완화
        under_relax(system, self.epsilon, self.alpha_eps)

        # 풀기
        self.epsilon.values = solve_linear_system(system, self.epsilon.values, method='bicgstab')
        self.epsilon.values = np.maximum(self.epsilon.values, 1e-10)

    def _update_mu_t(self):
        """난류 점성 갱신: μ_t = ρ·C_μ·k²/ε"""
        k = np.maximum(self.k.values, 1e-10)
        eps = np.maximum(self.epsilon.values, 1e-10)
        self.mu_t.values = self.rho * C_MU * k**2 / eps

    def _apply_wall_functions(self, U: VectorField):
        """
        벽함수 적용.

        벽 인접 셀에서의 k, ε 값을 벽함수로 보정한다.
        """
        mesh = self.mesh

        for patch_name in self.wall_patches:
            if patch_name not in mesh.boundary_patches:
                continue

            for fid in mesh.boundary_patches[patch_name]:
                face = mesh.faces[fid]
                ci = face.owner
                cell_center = mesh.cells[ci].center
                y_plus_dist = np.linalg.norm(cell_center - face.center)

                if y_plus_dist < 1e-15:
                    continue

                U_cell = np.linalg.norm(U.values[ci])
                if U_cell < 1e-15:
                    continue

                # 마찰 속도 추정
                u_tau = max(C_MU**0.25 * np.sqrt(max(self.k.values[ci], 1e-10)), 1e-10)
                y_plus = self.rho * u_tau * y_plus_dist / self.mu

                if y_plus > 11.225:
                    # 로그 영역: 벽함수 적용
                    self.k.values[ci] = max(u_tau**2 / np.sqrt(C_MU), 1e-10)
                    self.epsilon.values[ci] = max(u_tau**3 / (KAPPA * y_plus_dist), 1e-10)

    def get_turbulent_viscosity(self) -> ScalarField:
        """난류 점성 필드 반환."""
        return self.mu_t
