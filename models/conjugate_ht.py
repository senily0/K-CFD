"""
Conjugate Heat Transfer (CHT) 커플링.

유체-고체 인터페이스에서의 온도 연속 및 열유속 연속 조건을 적용한다.

인터페이스 조건:
  T_f|_interface = T_s|_interface        (온도 연속)
  k_f·(∂T_f/∂n) = k_s·(∂T_s/∂n)        (열유속 연속)
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField
from models.single_phase import SIMPLESolver
from models.solid_conduction import SolidConductionSolver


class CHTCoupling:
    """
    유체-고체 Conjugate Heat Transfer 커플링 관리자.

    유체 솔버와 고체 솔버 사이의 인터페이스 열전달을 처리한다.
    """

    def __init__(self, mesh: FVMesh, fluid_cells: list, solid_cells: list):
        self.mesh = mesh
        self.fluid_cells = set(fluid_cells)
        self.solid_cells = set(solid_cells)

        # 유체 솔버
        self.fluid_solver = SIMPLESolver(mesh)

        # 고체 솔버
        self.solid_solver = SolidConductionSolver(mesh, solid_cells)

        # 유체 온도장
        self.T_fluid = ScalarField(mesh, "T_fluid")
        self.T_fluid.set_uniform(300.0)

        # 유체 물성
        self.rho_f = 998.2
        self.cp_f = 4182.0
        self.k_f = 0.6
        self.mu_f = 1.003e-3

        # 인터페이스 면 식별
        self.interface_faces: list = []  # (face_id, fluid_cell, solid_cell)
        self._find_interface_faces()

        # CHT 설정
        self.max_cht_iter = 50
        self.tol_cht = 1e-5
        self.alpha_cht = 0.5  # CHT 완화계수

    def _find_interface_faces(self):
        """유체-고체 인터페이스 면 식별."""
        self.interface_faces = []
        for fid, face in enumerate(self.mesh.faces):
            if face.neighbour < 0:
                continue
            owner = face.owner
            nb = face.neighbour
            if (owner in self.fluid_cells and nb in self.solid_cells):
                self.interface_faces.append((fid, owner, nb))
            elif (owner in self.solid_cells and nb in self.fluid_cells):
                self.interface_faces.append((fid, nb, owner))

    def solve_steady(self) -> dict:
        """
        정상 CHT 해석.

        유체 유동 → 유체 에너지 → 고체 열전도를 반복하여 수렴시킨다.

        Returns
        -------
        result : {'converged': bool, 'iterations': int,
                  'T_interface_avg': float, 'heat_flux': float}
        """
        # 1) 유체 유동 먼저 풀기
        print("  [CHT] 유체 유동 해석 중...")
        flow_result = self.fluid_solver.solve_steady()
        print(f"  [CHT] 유동 수렴: {flow_result['converged']}, "
              f"반복: {flow_result['iterations']}")

        # 2) CHT 반복
        for cht_iter in range(self.max_cht_iter):
            T_interface_old = self._get_interface_temperatures()

            # 유체 에너지 방정식 풀기
            self._solve_fluid_energy()

            # 인터페이스 열유속 계산
            q_interface = self._compute_interface_heat_flux()

            # 고체 열전도 풀기 (인터페이스 열유속을 경계조건으로)
            self._apply_interface_bc_to_solid(q_interface)
            self.solid_solver.solve_steady()

            # 인터페이스 온도 갱신 및 유체 경계조건 갱신
            T_interface_new = self._get_interface_temperatures()
            self._apply_interface_bc_to_fluid(T_interface_new)

            # 수렴 판정
            if len(T_interface_old) > 0 and len(T_interface_new) > 0:
                change = np.linalg.norm(
                    np.array(T_interface_new) - np.array(T_interface_old)
                )
                norm = max(np.linalg.norm(np.array(T_interface_new)), 1e-15)
                res = change / norm

                if res < self.tol_cht:
                    T_avg = np.mean(T_interface_new) if T_interface_new else 300.0
                    q_avg = np.mean(np.abs(q_interface)) if len(q_interface) > 0 else 0.0
                    return {
                        'converged': True,
                        'iterations': cht_iter + 1,
                        'T_interface_avg': float(T_avg),
                        'heat_flux': float(q_avg)
                    }

        T_interface = self._get_interface_temperatures()
        T_avg = np.mean(T_interface) if T_interface else 300.0
        return {
            'converged': False,
            'iterations': self.max_cht_iter,
            'T_interface_avg': float(T_avg),
            'heat_flux': 0.0
        }

    def _solve_fluid_energy(self):
        """
        유체 에너지 방정식 (간단한 대류-확산).

        ρ·cp·∇·(u·T) = ∇·(k·∇T)
        """
        from core.fvm_operators import (FVMSystem, diffusion_operator,
                                         convection_operator_upwind,
                                         apply_boundary_conditions, under_relax)
        from core.interpolation import compute_mass_flux
        from core.linear_solver import solve_linear_system

        mesh = self.mesh
        n = mesh.n_cells
        system = FVMSystem(n)

        # 확산 계수
        gamma = ScalarField(mesh, "k_fluid")
        gamma.set_uniform(self.k_f)

        # 확산
        diffusion_operator(mesh, gamma, system)

        # 대류
        mass_flux_T = compute_mass_flux(self.fluid_solver.U, self.rho_f, mesh)
        # cp를 곱한 질량유속
        mass_flux_T *= self.cp_f
        convection_operator_upwind(mesh, mass_flux_T, system)

        # 경계조건
        bc_T = {}
        for bname in mesh.boundary_patches:
            if 'inlet' in bname:
                bc_T[bname] = {'type': 'dirichlet'}
                self.T_fluid.set_boundary(bname, 300.0)
            elif 'outlet' in bname:
                bc_T[bname] = {'type': 'zero_gradient'}
            else:
                bc_T[bname] = {'type': 'zero_gradient'}

        apply_boundary_conditions(mesh, self.T_fluid, gamma,
                                 mass_flux_T, system, bc_T)

        under_relax(system, self.T_fluid, 0.8)

        self.T_fluid.values = solve_linear_system(system, self.T_fluid.values,
                                                    method='bicgstab')

    def _compute_interface_heat_flux(self) -> np.ndarray:
        """인터페이스 면에서의 열유속 계산."""
        q_list = []
        for fid, fc, sc in self.interface_faces:
            face = self.mesh.faces[fid]
            T_f = self.T_fluid.values[fc]
            T_s = self.solid_solver.T.values[sc]
            d = np.linalg.norm(
                self.mesh.cells[fc].center - self.mesh.cells[sc].center
            )
            if d < 1e-30:
                q_list.append(0.0)
                continue

            # 유효 열전달 계수 (조화 평균)
            k_eff = 2.0 * self.k_f * self.solid_solver.k_s / (
                self.k_f + self.solid_solver.k_s
            )
            q = k_eff * (T_s - T_f) / d
            q_list.append(q)

        return np.array(q_list)

    def _get_interface_temperatures(self) -> list:
        """인터페이스 면에서의 온도 리스트."""
        temps = []
        for fid, fc, sc in self.interface_faces:
            T_f = self.T_fluid.values[fc]
            T_s = self.solid_solver.T.values[sc]
            # 연속 인터페이스 온도
            k_eff_f = self.k_f
            k_eff_s = self.solid_solver.k_s
            T_int = (k_eff_f * T_f + k_eff_s * T_s) / (k_eff_f + k_eff_s)
            temps.append(T_int)
        return temps

    def _apply_interface_bc_to_solid(self, q_interface: np.ndarray):
        """고체 솔버에 인터페이스 열유속 적용."""
        # 인터페이스 인접 고체 셀에 소스로 적용
        for idx, (fid, fc, sc) in enumerate(self.interface_faces):
            face = self.mesh.faces[fid]
            vol = self.mesh.cells[sc].volume
            if vol > 1e-30:
                self.solid_solver.q_vol[sc] = q_interface[idx] * face.area / vol

    def _apply_interface_bc_to_fluid(self, T_interface: list):
        """유체 솔버에 인터페이스 온도 적용."""
        for idx, (fid, fc, sc) in enumerate(self.interface_faces):
            if idx < len(T_interface):
                # 인터페이스 인접 유체 셀에 고정 온도 효과
                T_int = T_interface[idx]
                self.T_fluid.values[fc] = (
                    self.alpha_cht * T_int +
                    (1.0 - self.alpha_cht) * self.T_fluid.values[fc]
                )

    def get_interface_data(self) -> dict:
        """인터페이스 데이터 반환 (시각화용)."""
        x_coords = []
        T_fluid_vals = []
        T_solid_vals = []
        T_interface_vals = []

        for fid, fc, sc in self.interface_faces:
            face = self.mesh.faces[fid]
            x_coords.append(face.center[0])
            T_fluid_vals.append(self.T_fluid.values[fc])
            T_solid_vals.append(self.solid_solver.T.values[sc])
            T_int = self._get_interface_temperatures()
            if T_int:
                idx = len(x_coords) - 1
                if idx < len(T_int):
                    T_interface_vals.append(T_int[idx])

        idx = np.argsort(x_coords) if x_coords else []
        return {
            'x': np.array(x_coords)[idx] if len(idx) > 0 else np.array([]),
            'T_fluid': np.array(T_fluid_vals)[idx] if len(idx) > 0 else np.array([]),
            'T_solid': np.array(T_solid_vals)[idx] if len(idx) > 0 else np.array([]),
            'T_interface': np.array(T_interface_vals)[idx] if len(idx) > 0 else np.array([])
        }
