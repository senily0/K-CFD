"""
Two-Fluid Model (Euler-Euler) 솔버.

각 상(액체/기체)에 대한 연속, 운동량, 에너지 방정식을 풀며,
SIMPLE 알고리즘으로 속도-압력 커플링을 처리한다.
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.time_control import AdaptiveTimeControl
from core.fields import ScalarField, VectorField
from core.fvm_operators import (FVMSystem, diffusion_operator,
                                 convection_operator_upwind, temporal_operator,
                                 linearized_source, apply_boundary_conditions,
                                 under_relax)
from core.interpolation import compute_mass_flux
from core.linear_solver import solve_linear_system, solve_pressure_correction
from models.closure import (drag_coefficient_implicit, sato_bubble_induced_turbulence,
                             interfacial_heat_transfer)


class TwoFluidSolver:
    """
    Euler-Euler Two-Fluid Model 솔버.

    액체상(l)과 기체상(g)에 대한 연립 방정식 시스템.
    6-equation 모드: 각 상별 에너지 방정식을 SIMPLE 내부에서 암시적으로 풀어
    비평형 상변화를 지원한다.
    """

    def __init__(self, mesh: FVMesh):
        self.mesh = mesh
        n = mesh.n_cells

        # 물성치 (물-공기 기본값)
        self.rho_l = 998.2     # 액체 밀도 [kg/m³]
        self.rho_g = 1.225     # 기체 밀도 [kg/m³]
        self.mu_l = 1.003e-3   # 액체 점성 [Pa·s]
        self.mu_g = 1.789e-5   # 기체 점성 [Pa·s]
        self.cp_l = 4182.0     # 액체 비열 [J/(kg·K)]
        self.cp_g = 1006.0     # 기체 비열 [J/(kg·K)]
        self.k_l = 0.6         # 액체 열전도도 [W/(m·K)]
        self.k_g = 0.0257      # 기체 열전도도 [W/(m·K)]
        self.d_b = 0.005       # 기포 직경 [m]
        ndim = getattr(mesh, 'ndim', 2)
        self.g = np.zeros(ndim)
        self.g[-1] = -9.81  # 중력 (마지막 축 방향)

        # 6-equation 모델 파라미터
        self.h_i_coeff = 1.0   # 상간 열전달 계수 스케일 팩터
        self.a_i_coeff = 1.0   # 비계면적 스케일 팩터
        self.T_sat = 373.15    # 포화 온도 [K]
        self.h_fg = 2.257e6    # 증발 잠열 [J/kg]
        self.r_phase_change = 0.1  # 상변화 계수 [1/s]
        self.alpha_T = 0.7     # 온도 완화계수

        # 필드
        self.alpha_l = ScalarField(mesh, "alpha_liquid")
        self.alpha_g = ScalarField(mesh, "alpha_gas")
        self.U_l = VectorField(mesh, "velocity_liquid")
        self.U_g = VectorField(mesh, "velocity_gas")
        self.p = ScalarField(mesh, "pressure")
        self.T_l = ScalarField(mesh, "temperature_liquid")
        self.T_g = ScalarField(mesh, "temperature_gas")

        # 초기 체적분율
        self.alpha_l.set_uniform(0.95)
        self.alpha_g.set_uniform(0.05)

        # 온도 초기값
        self.T_l.set_uniform(300.0)
        self.T_g.set_uniform(300.0)

        # 솔버 설정
        self.alpha_u = 0.5    # 속도 완화계수
        self.alpha_p = 0.3    # 압력 완화계수
        self.alpha_alpha = 0.5  # 체적분율 완화계수
        self.max_outer_iter = 200
        self.tol = 1e-4
        self.dt = 0.001
        self.solve_energy = False
        self.solve_momentum = True  # False이면 속도장 고정 (에너지만 해석)

        # 벽면 열유속 저장 {patch_name: q_wall [W/m²]}
        self.wall_heat_flux: dict = {}

        # 경계조건
        self.bc_alpha: dict = {}
        self.bc_u_l: dict = {}
        self.bc_u_g: dict = {}
        self.bc_p: dict = {}
        self.bc_T_l: dict = {}
        self.bc_T_g: dict = {}

        # 잔차
        self.residuals: list = []
        self.energy_residuals: list = []

    def initialize(self, alpha_g_init: float = 0.05):
        """체적분율 초기화."""
        self.alpha_g.set_uniform(alpha_g_init)
        self.alpha_l.set_uniform(1.0 - alpha_g_init)

    def set_inlet_bc(self, patch: str, alpha_g: float, U_l: list, U_g: list,
                     T_l: float = None, T_g: float = None):
        """입구 경계조건."""
        self.alpha_g.set_boundary(patch, alpha_g)
        self.alpha_l.set_boundary(patch, 1.0 - alpha_g)
        self.U_l.set_boundary(patch, U_l)
        self.U_g.set_boundary(patch, U_g)

        self.bc_alpha[patch] = {'type': 'dirichlet'}
        self.bc_u_l[patch] = {'type': 'dirichlet'}
        self.bc_u_g[patch] = {'type': 'dirichlet'}
        self.bc_p[patch] = {'type': 'zero_gradient'}

        # 온도 경계조건
        if T_l is not None:
            self.T_l.set_boundary(patch, T_l)
            self.bc_T_l[patch] = {'type': 'dirichlet'}
        if T_g is not None:
            self.T_g.set_boundary(patch, T_g)
            self.bc_T_g[patch] = {'type': 'dirichlet'}

    def set_outlet_bc(self, patch: str, p_val: float = 0.0):
        """출구 경계조건."""
        self.p.set_boundary(patch, p_val)
        self.bc_p[patch] = {'type': 'dirichlet'}
        self.bc_u_l[patch] = {'type': 'zero_gradient'}
        self.bc_u_g[patch] = {'type': 'zero_gradient'}
        self.bc_alpha[patch] = {'type': 'zero_gradient'}
        self.bc_T_l[patch] = {'type': 'zero_gradient'}
        self.bc_T_g[patch] = {'type': 'zero_gradient'}

    def set_wall_bc(self, patch: str, q_wall: float = None, T_wall: float = None):
        """
        벽면 경계조건 (no-slip 액체, free-slip 기체).

        Parameters
        ----------
        patch : 경계 이름
        q_wall : 벽면 열유속 [W/m²] (양수 = 유체로 열 유입)
        T_wall : 벽면 고정 온도 [K]
        """
        ndim = getattr(self.mesh, 'ndim', 2)
        self.U_l.set_boundary(patch, np.zeros(ndim))
        self.bc_u_l[patch] = {'type': 'dirichlet'}
        # 기체: free-slip
        self.bc_u_g[patch] = {'type': 'zero_gradient'}
        self.bc_p[patch] = {'type': 'zero_gradient'}
        self.bc_alpha[patch] = {'type': 'zero_gradient'}

        # 온도 경계조건
        if q_wall is not None:
            self.wall_heat_flux[patch] = q_wall
            self.bc_T_l[patch] = {'type': 'zero_gradient'}
            self.bc_T_g[patch] = {'type': 'zero_gradient'}
        elif T_wall is not None:
            self.T_l.set_boundary(patch, T_wall)
            self.T_g.set_boundary(patch, T_wall)
            self.bc_T_l[patch] = {'type': 'dirichlet'}
            self.bc_T_g[patch] = {'type': 'dirichlet'}
        else:
            self.bc_T_l[patch] = {'type': 'zero_gradient'}
            self.bc_T_g[patch] = {'type': 'zero_gradient'}

    def solve_transient(self, t_end: float, dt: float = None,
                        report_interval: int = 100,
                        adaptive_dt: bool = False, cfl_target: float = 0.5,
                        dt_min: float = 1e-8, dt_max: float = None,
                        monitor=None) -> dict:
        """
        비정상 Two-Fluid 해석.

        Parameters
        ----------
        t_end : 최종 시간
        dt : 시간 간격
        report_interval : 잔차 출력 간격
        adaptive_dt : CFL 기반 적응 시간 간격 사용 여부
        cfl_target : 목표 CFL 수 (adaptive_dt=True 시)
        dt_min : 최소 시간 간격
        dt_max : 최대 시간 간격 (None이면 t_end 사용)
        monitor : update(step, t, residual, dt=dt) 콜백 객체
        """
        if dt is not None:
            self.dt = dt

        # Adaptive time control setup
        time_ctrl = None
        if adaptive_dt:
            _dt_max = dt_max if dt_max is not None else t_end
            time_ctrl = AdaptiveTimeControl(
                dt_init=self.dt,
                dt_min=dt_min,
                dt_max=_dt_max,
                cfl_target=cfl_target,
            )

        t = 0.0
        step = 0
        self.residuals = []
        dt_history = []
        cfl_history = []

        while t < t_end - 1e-15:
            # 이전 값 저장
            self.alpha_g.store_old()
            self.alpha_l.store_old()
            self.U_l.store_old()
            self.U_g.store_old()
            if self.solve_energy:
                self.T_l.store_old()
                self.T_g.store_old()

            # SIMPLE 내부 반복
            max_res = self._simple_iteration()

            # Adaptive time stepping
            if adaptive_dt and time_ctrl is not None:
                vel = self.U_l.values  # (n_cells, ndim)
                alpha_diff = self.mu_l / self.rho_l
                new_dt, dt_info = time_ctrl.compute_dt(
                    self.mesh, vel, alpha_diff,
                    converged=(max_res < 10 * self.tol)
                )
                self.dt = new_dt
                dt_history.append(new_dt)
                cfl_history.append(dt_info['cfl_max'])

            t += self.dt
            step += 1
            self.residuals.append(max_res)

            # Monitor callback
            if monitor is not None:
                monitor.update(step, t, max_res, dt=self.dt)

            if step % report_interval == 0:
                print(f"  Step {step}, t={t:.4f}, residual={max_res:.2e}, "
                      f"alpha_g: [{self.alpha_g.min():.4f}, {self.alpha_g.max():.4f}]")

        return {
            'time_steps': step,
            'final_time': t,
            'residuals': self.residuals,
            'dt_history': dt_history,
            'cfl_history': cfl_history,
        }

    def _update_zero_gradient_boundaries(self):
        """zero_gradient 경계의 boundary_values를 인접 셀값으로 갱신."""
        mesh = self.mesh
        # 속도
        for bc_dict, field in [(self.bc_u_l, self.U_l), (self.bc_u_g, self.U_g)]:
            for bname, bc in bc_dict.items():
                if bc['type'] == 'zero_gradient' and bname in mesh.boundary_patches:
                    for local_idx, fid in enumerate(mesh.boundary_patches[bname]):
                        owner = mesh.faces[fid].owner
                        field.boundary_values[bname][local_idx] = field.values[owner]
        # 온도
        if self.solve_energy:
            for bc_dict, field in [(self.bc_T_l, self.T_l), (self.bc_T_g, self.T_g)]:
                for bname, bc in bc_dict.items():
                    if bc['type'] == 'zero_gradient' and bname in mesh.boundary_patches:
                        for local_idx, fid in enumerate(mesh.boundary_patches[bname]):
                            owner = mesh.faces[fid].owner
                            field.boundary_values[bname][local_idx] = field.values[owner]
        # 체적분율
        for bname, bc in self.bc_alpha.items():
            if bc['type'] == 'zero_gradient' and bname in mesh.boundary_patches:
                for local_idx, fid in enumerate(mesh.boundary_patches[bname]):
                    owner = mesh.faces[fid].owner
                    self.alpha_g.boundary_values[bname][local_idx] = self.alpha_g.values[owner]
                    self.alpha_l.boundary_values[bname][local_idx] = self.alpha_l.values[owner]

    def _simple_iteration(self) -> float:
        """한 시간 스텝에서의 SIMPLE 반복."""
        mesh = self.mesh

        n_inner = min(self.max_outer_iter, 30)
        for inner in range(n_inner):
            # zero_gradient 경계값 갱신
            self._update_zero_gradient_boundaries()
            # 질량유속
            mf_l = compute_mass_flux(self.U_l, self.rho_l, mesh)
            mf_g = compute_mass_flux(self.U_g, self.rho_g, mesh)

            res_mom = []
            res_p = 0.0
            res_alpha = 0.0

            if self.solve_momentum:
                # 항력 계수
                K_drag = drag_coefficient_implicit(
                    self.alpha_g.values, self.rho_l,
                    self.U_g.values, self.U_l.values,
                    self.d_b, self.mu_l
                )

                # 운동량 방정식 (각 상, 각 성분)
                ndim = getattr(mesh, 'ndim', 2)
                for comp in range(ndim):
                    res_mom.append(self._solve_phase_momentum(
                        'liquid', comp, mf_l, K_drag))
                for comp in range(ndim):
                    res_mom.append(self._solve_phase_momentum(
                        'gas', comp, mf_g, K_drag))

                # 압력 보정
                res_p = self._solve_pressure_correction_two_phase(mf_l, mf_g)

                # 체적분율 방정식 (상변화 소스항 포함)
                dot_m = self._compute_phase_change_rate() if self.solve_energy else np.zeros(mesh.n_cells)
                res_alpha = self._solve_volume_fraction(mf_g, dot_m)
            else:
                # 속도장 고정: 에너지 및 체적분율 해석
                dot_m = self._compute_phase_change_rate() if self.solve_energy else np.zeros(mesh.n_cells)
                if self.solve_energy:
                    res_alpha = self._solve_volume_fraction(mf_g, dot_m)

            # 6-equation 에너지: SIMPLE 내부에서 암시적으로 풀기
            res_energy = 0.0
            if self.solve_energy:
                res_energy = self._solve_coupled_energy(mf_l, mf_g, dot_m)

            all_res = res_mom + [res_p, res_alpha, res_energy]
            max_res = max(r if np.isfinite(r) else 1e10 for r in all_res)
            if max_res < self.tol:
                break

        return max_res if np.isfinite(max_res) else 1e10

    def _compute_phase_change_rate(self) -> np.ndarray:
        """
        상변화 질량 전달률 계산.

        dot_m > 0: 증발 (액체 → 기체)
        dot_m < 0: 응축 (기체 → 액체)
        """
        n = self.mesh.n_cells
        dot_m = np.zeros(n)

        # Lee 모델 기반 상변화율
        r = self.r_phase_change
        for ci in range(n):
            T_l = self.T_l.values[ci]
            T_g = self.T_g.values[ci]
            al = self.alpha_l.values[ci]
            ag = self.alpha_g.values[ci]

            # 증발: T_l > T_sat
            if T_l > self.T_sat and al > 1e-6:
                dot_m[ci] += r * al * self.rho_l * (T_l - self.T_sat) / self.T_sat

            # 응축: T_g < T_sat
            if T_g < self.T_sat and ag > 1e-6:
                dot_m[ci] -= r * ag * self.rho_g * (self.T_sat - T_g) / self.T_sat

        return dot_m

    def _solve_coupled_energy(self, mf_l: np.ndarray, mf_g: np.ndarray,
                               dot_m: np.ndarray) -> float:
        """
        6-equation 에너지: 각 상별 에너지 방정식을 암시적으로 풀기.

        상간 열전달 Q_i = h_i · a_i · (T_g - T_l) 선형화:
          액상: Su += h_i*a_i*T_g, Sp += -h_i*a_i
          기상: Su += h_i*a_i*T_l, Sp += -h_i*a_i
        """
        mesh = self.mesh
        n = mesh.n_cells

        # 상간 열전달 계수 계산
        h_i, a_i = self._compute_interfacial_ht_coefficients()

        # 액상 에너지 방정식
        res_T_l = self._solve_phase_energy(
            'liquid', self.dt,
            self.alpha_l, self.rho_l, self.cp_l, self.k_l,
            self.U_l, mf_l, self.T_l, self.T_g,
            h_i, a_i, dot_m, self.bc_T_l)

        # 기상 에너지 방정식
        res_T_g = self._solve_phase_energy(
            'gas', self.dt,
            self.alpha_g, self.rho_g, self.cp_g, self.k_g,
            self.U_g, mf_g, self.T_g, self.T_l,
            h_i, a_i, -dot_m, self.bc_T_g)

        self.energy_residuals.append(max(res_T_l, res_T_g))
        return max(res_T_l, res_T_g)

    def _compute_interfacial_ht_coefficients(self):
        """상간 열전달 계수 h_i, a_i 계산 (Ranz-Marshall)."""
        n = self.mesh.n_cells
        u_rel = self.U_g.values - self.U_l.values
        u_rel_mag = np.sqrt(np.sum(u_rel**2, axis=1))
        u_rel_mag = np.maximum(u_rel_mag, 1e-15)

        Re_p = self.rho_l * u_rel_mag * self.d_b / self.mu_l
        Pr = self.mu_l * self.cp_l / self.k_l
        Nu = 2.0 + 0.6 * np.maximum(Re_p, 1e-10)**0.5 * Pr**0.333

        h_i = self.h_i_coeff * Nu * self.k_l / self.d_b
        a_i = self.a_i_coeff * 6.0 * self.alpha_g.values / np.maximum(self.d_b, 1e-15)

        return h_i, a_i

    def _solve_phase_energy(self, phase: str, dt: float,
                             alpha: ScalarField, rho: float,
                             cp: float, k_cond: float,
                             U: VectorField, mass_flux: np.ndarray,
                             T: ScalarField, T_other: ScalarField,
                             h_i: np.ndarray, a_i: np.ndarray,
                             dot_m: np.ndarray,
                             bc_T: dict) -> float:
        """
        상별 에너지 방정식 (운동량 방정식 패턴 복제).

        ∂(α_k·ρ_k·cp_k·T_k)/∂t + ∇·(α_k·ρ_k·cp_k·U_k·T_k)
            = ∇·(α_k·k_k·∇T_k) + Q_interfacial + Q_phase_change + Q_wall
        """
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        # 확산 (α·k_thermal)
        gamma = ScalarField(mesh, f"gamma_T_{phase}")
        gamma.values = alpha.values * k_cond

        diffusion_operator(mesh, gamma, system)

        # 대류 (α·ρ·cp·U): 면에서 alpha를 upwind 보간하여 사용
        alpha_face = np.zeros(mesh.n_faces)
        for fid, face in enumerate(mesh.faces):
            if face.neighbour >= 0:
                F = mass_flux[fid]
                if F >= 0:
                    alpha_face[fid] = alpha.values[face.owner]
                else:
                    alpha_face[fid] = alpha.values[face.neighbour]
            else:
                alpha_face[fid] = alpha.values[face.owner]
        alpha_cp_mf = mass_flux * alpha_face * cp
        convection_operator_upwind(mesh, alpha_cp_mf, system)

        # 시간항
        if T.old_values is not None:
            for ci in range(n):
                vol = mesh.cells[ci].volume
                coeff = alpha.values[ci] * rho * cp * vol / dt
                system.add_diagonal(ci, coeff)
                system.add_source(ci, coeff * T.old_values[ci])

        # 상간 열전달 선형화: Q_i = h_i*a_i*(T_other - T)
        #   Su += h_i*a_i*T_other, Sp += -h_i*a_i
        for ci in range(n):
            vol = mesh.cells[ci].volume
            hi_ai = h_i[ci] * a_i[ci] * vol
            system.add_source(ci, hi_ai * T_other.values[ci])
            system.add_diagonal(ci, hi_ai)

        # 상변화 잠열: dot_m > 0 증발
        for ci in range(n):
            vol = mesh.cells[ci].volume
            if phase == 'liquid':
                # 액상: 증발 시 잠열 흡수 (냉각)
                system.add_source(ci, -dot_m[ci] * self.h_fg * vol)
            else:
                # 기상: 증발 시 잠열 방출? 아니라, 증발 생성물이 포화온도로 유입
                # dot_m은 이미 부호가 반전된 상태 (-dot_m)
                system.add_source(ci, -dot_m[ci] * self.h_fg * vol)

        # 벽면 열유속 소스항
        # 열유속은 해당 상의 체적분율에 비례하여 분배
        # 액상: q_wall * alpha_l / (alpha_l + alpha_g_eps) → 주로 액상이 벽에서 열 흡수
        for patch_name, q_wall in self.wall_heat_flux.items():
            if patch_name in mesh.boundary_patches:
                for fid in mesh.boundary_patches[patch_name]:
                    face = mesh.faces[fid]
                    owner = face.owner
                    al = self.alpha_l.values[owner]
                    ag = self.alpha_g.values[owner]
                    # 벽면 열유속을 체적분율로 가중: 액상은 al/(al+ag)=al 비율로 흡수
                    # (al + ag = 1 이므로 단순히 alpha로 가중)
                    if phase == 'liquid':
                        system.add_source(owner, q_wall * face.area * al)
                    else:
                        system.add_source(owner, q_wall * face.area * ag)

        # 경계조건 (대류 유속은 alpha*cp*mf 사용 — convection_operator와 일치)
        apply_boundary_conditions(mesh, T, gamma, alpha_cp_mf, system, bc_T)

        # 대각선 최소값 보장 (특이 행렬 방지)
        for ci in range(mesh.n_cells):
            if system.diag[ci] < 1e-20:
                system.diag[ci] = 1e-10

        # 완화
        under_relax(system, T, self.alpha_T)

        # 풀기
        T_old = T.values.copy()
        T.values = solve_linear_system(system, T.values, method='direct')

        # NaN 보호 및 온도 제한
        if np.any(np.isnan(T.values)) or np.any(np.isinf(T.values)):
            T.values = T_old
        else:
            # 물리적 온도 범위 제한 (200K ~ 800K)
            T.values = np.clip(T.values, 200.0, 800.0)

        # 잔차 계산
        A = system.to_sparse()
        residual = np.linalg.norm(A @ T.values - system.rhs)
        norm = max(np.linalg.norm(system.rhs), 1e-15)
        return residual / norm

    def _solve_phase_momentum(self, phase: str, comp: int,
                               mass_flux: np.ndarray,
                               K_drag: np.ndarray) -> float:
        """상별 운동량 방정식 풀기."""
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        if phase == 'liquid':
            U = self.U_l
            alpha = self.alpha_l
            rho = self.rho_l
            mu = self.mu_l
            U_other = self.U_g
            bc = self.bc_u_l
        else:
            U = self.U_g
            alpha = self.alpha_g
            rho = self.rho_g
            mu = self.mu_g
            U_other = self.U_l
            bc = self.bc_u_g

        phi = ScalarField(mesh, f"u_{phase}_{comp}")
        phi.values = U.values[:, comp].copy()
        for bname in U.boundary_values:
            phi.boundary_values[bname] = U.boundary_values[bname][:, comp].copy()
        if U.old_values is not None:
            phi.old_values = U.old_values[:, comp].copy()

        # 확산 (α·μ)
        gamma = ScalarField(mesh, "gamma")
        gamma.values = alpha.values * mu

        # BIT 추가
        mu_BIT = sato_bubble_induced_turbulence(
            self.alpha_g.values, self.rho_l,
            self.U_g.values, self.U_l.values, self.d_b
        )
        if phase == 'liquid':
            gamma.values += alpha.values * mu_BIT

        diffusion_operator(mesh, gamma, system)

        # 대류 (α·ρ·u)
        alpha_mf = mass_flux * alpha.values[np.array([
            mesh.faces[f].owner for f in range(mesh.n_faces)
        ])]
        convection_operator_upwind(mesh, alpha_mf, system)

        # 시간항
        if phi.old_values is not None:
            for ci in range(n):
                vol = mesh.cells[ci].volume
                coeff = alpha.values[ci] * rho * vol / self.dt
                system.add_diagonal(ci, coeff)
                system.add_source(ci, coeff * phi.old_values[ci])

        # 압력 기울기: -α·∇p
        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            if face.neighbour >= 0:
                p_f = 0.5 * (self.p.values[owner] + self.p.values[face.neighbour])
            else:
                p_f = self.p.values[owner]
            force = -alpha.values[owner] * p_f * face.normal[comp] * face.area
            system.add_source(owner, force)
            if face.neighbour >= 0:
                nb = face.neighbour
                force_nb = -alpha.values[nb] * p_f * face.normal[comp] * face.area
                system.add_source(nb, -force_nb)

        # 항력 (암시적)
        for ci in range(n):
            vol = mesh.cells[ci].volume
            system.add_diagonal(ci, K_drag[ci] * vol)
            system.add_source(ci, K_drag[ci] * vol * U_other.values[ci, comp])

        # 중력
        for ci in range(n):
            vol = mesh.cells[ci].volume
            system.add_source(ci, alpha.values[ci] * rho * self.g[comp] * vol)

        # 경계조건 (대류 유속은 alpha*mf 사용 — convection_operator와 일치)
        apply_boundary_conditions(mesh, phi, gamma, alpha_mf, system, bc)

        # 대각선 최소값 보장 (특이 행렬 방지)
        for ci in range(n):
            if system.diag[ci] < 1e-20:
                system.diag[ci] = 1e-10

        # 완화
        under_relax(system, phi, self.alpha_u)

        # 풀기
        phi.values = solve_linear_system(system, phi.values, method='direct')

        # NaN 보호: NaN 발생 시 이전 값 유지
        if np.any(np.isnan(phi.values)):
            phi.values = U.values[:, comp].copy()

        U.values[:, comp] = phi.values

        # a_P 저장 (모든 성분의 최대값 사용 — 주 유동 방향 반영)
        aP_cur = np.maximum(system.diag.copy(), 1e-20)
        if phase == 'liquid':
            if not hasattr(self, '_aP_l') or comp == 0:
                self._aP_l = aP_cur
            else:
                self._aP_l = np.maximum(self._aP_l, aP_cur)
        else:
            if not hasattr(self, '_aP_g') or comp == 0:
                self._aP_g = aP_cur
            else:
                self._aP_g = np.maximum(self._aP_g, aP_cur)

        A = system.to_sparse()
        residual = np.linalg.norm(A @ phi.values - system.rhs)
        norm = max(np.linalg.norm(system.rhs), 1e-15)
        return residual / norm

    def _solve_pressure_correction_two_phase(self, mf_l: np.ndarray,
                                              mf_g: np.ndarray) -> float:
        """2상 압력 보정 방정식."""
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        if not hasattr(self, '_aP_l'):
            self._aP_l = np.ones(n)
            self._aP_g = np.ones(n)

        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            if face.neighbour >= 0:
                nb = face.neighbour
                d_PN = np.linalg.norm(
                    mesh.cells[nb].center - mesh.cells[owner].center
                )
                if d_PN < 1e-30:
                    continue

                # 양상 기여
                al_f = 0.5 * (self.alpha_l.values[owner] + self.alpha_l.values[nb])
                ag_f = 0.5 * (self.alpha_g.values[owner] + self.alpha_g.values[nb])
                aP_l_f = 0.5 * (self._aP_l[owner] + self._aP_l[nb])
                aP_g_f = 0.5 * (self._aP_g[owner] + self._aP_g[nb])

                vol_f = 0.5 * (mesh.cells[owner].volume + mesh.cells[nb].volume)

                coeff_l = self.rho_l * al_f * vol_f / max(aP_l_f, 1e-30)
                coeff_g = self.rho_g * ag_f * vol_f / max(aP_g_f, 1e-30)
                coeff = (coeff_l + coeff_g) * face.area / d_PN

                system.add_diagonal(owner, coeff)
                system.add_diagonal(nb, coeff)
                system.add_off_diagonal(owner, nb, -coeff)
                system.add_off_diagonal(nb, owner, -coeff)

                # 우변: 질량 불균형
                total_mf = mf_l[fid] * self.alpha_l.values[owner] + \
                           mf_g[fid] * self.alpha_g.values[owner]
                system.add_source(owner, -total_mf)
                system.add_source(nb, total_mf)
            else:
                total_mf = mf_l[fid] * self.alpha_l.values[owner] + \
                           mf_g[fid] * self.alpha_g.values[owner]
                system.add_source(owner, -total_mf)

        system.add_diagonal(0, 1e10)

        p_prime = solve_pressure_correction(system)

        # NaN 보호
        if np.any(np.isnan(p_prime)):
            p_prime = np.zeros_like(p_prime)

        self.p.values += self.alpha_p * p_prime

        # 속도 보정: u'_k = -(V/a_P_k) * ∇p' (셀 기반 근사)
        ndim = getattr(mesh, 'ndim', 2)
        grad_p_prime = np.zeros((n, ndim))
        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            if face.neighbour >= 0:
                nb = face.neighbour
                dp = p_prime[nb] - p_prime[owner]
                for d in range(ndim):
                    flux = dp * face.normal[d] * face.area
                    grad_p_prime[owner, d] += flux
                    grad_p_prime[nb, d] -= flux
            # 경계면: p' = 0 (Dirichlet 압력 경계) 또는 ∂p'/∂n = 0
        for ci in range(n):
            vol = mesh.cells[ci].volume
            if vol > 1e-30:
                grad_p_prime[ci] /= vol

        for ci in range(n):
            vol = mesh.cells[ci].volume
            for d in range(ndim):
                corr_l = -self.alpha_l.values[ci] * vol / max(self._aP_l[ci], 1e-20) * grad_p_prime[ci, d]
                corr_g = -self.alpha_g.values[ci] * vol / max(self._aP_g[ci], 1e-20) * grad_p_prime[ci, d]
                self.U_l.values[ci, d] += self.alpha_p * corr_l
                self.U_g.values[ci, d] += self.alpha_p * corr_g

        return float(np.linalg.norm(p_prime) / max(np.linalg.norm(self.p.values), 1e-15))

    def _solve_volume_fraction(self, mf_g: np.ndarray,
                                dot_m: np.ndarray = None) -> float:
        """
        기체 체적분율 수송 방정식.

        Parameters
        ----------
        mf_g : 기체 질량유속
        dot_m : 상변화 질량 전달률 (양수=증발, 액→기)
        """
        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        gamma_zero = ScalarField(mesh, "gamma_zero")
        gamma_zero.set_uniform(1e-6)

        convection_operator_upwind(mesh, mf_g, system)

        if self.alpha_g.old_values is not None:
            temporal_operator(mesh, self.rho_g, self.dt,
                            self.alpha_g.old_values, system)

        # 상변화 소스항: dot_m/rho_g (증발 시 기체 체적분율 증가)
        if dot_m is not None:
            for ci in range(n):
                vol = mesh.cells[ci].volume
                if abs(dot_m[ci]) > 1e-30:
                    system.add_source(ci, dot_m[ci] / self.rho_g * vol)

        apply_boundary_conditions(mesh, self.alpha_g, gamma_zero,
                                mf_g, system, self.bc_alpha)

        # 대각선 최소값 보장
        for ci in range(n):
            if system.diag[ci] < 1e-20:
                system.diag[ci] = 1e-10

        under_relax(system, self.alpha_g, self.alpha_alpha)

        alpha_old = self.alpha_g.values.copy()
        self.alpha_g.values = solve_linear_system(system, self.alpha_g.values,
                                                   method='direct')

        # NaN 보호
        if np.any(np.isnan(self.alpha_g.values)):
            self.alpha_g.values = alpha_old

        # 물리적 제한
        self.alpha_g.values = np.clip(self.alpha_g.values, 0.0, 1.0)
        self.alpha_l.values = 1.0 - self.alpha_g.values

        return float(np.max(np.abs(
            self.alpha_g.values - (self.alpha_g.old_values
                                    if self.alpha_g.old_values is not None
                                    else self.alpha_g.values)
        )))

    def _solve_energy(self):
        """에너지 방정식 (하위 호환: solve_energy=True이면 SIMPLE 내부에서 처리됨)."""
        # 6-equation 모델에서는 _solve_coupled_energy()가 SIMPLE 내부에서 호출됨.
        # 이 메서드는 하위 호환용 폴백으로, SIMPLE 외부 호출 시에만 사용.
        mesh = self.mesh
        n = mesh.n_cells

        Q_interf = interfacial_heat_transfer(
            self.alpha_g.values, self.rho_l,
            self.U_g.values, self.U_l.values,
            self.T_g.values, self.T_l.values,
            self.d_b, self.mu_l, self.cp_l, self.k_l
        )

        for ci in range(n):
            al = self.alpha_l.values[ci]
            ag = self.alpha_g.values[ci]
            if al * self.rho_l * self.cp_l > 1e-15:
                self.T_l.values[ci] += self.dt * Q_interf[ci] / (al * self.rho_l * self.cp_l)
            if ag * self.rho_g * self.cp_g > 1e-15:
                self.T_g.values[ci] -= self.dt * Q_interf[ci] / (ag * self.rho_g * self.cp_g)
