"""
FVM 이산화 연산자: 확산, 대류, 소스, 시간항.

선형 시스템 Ax = b 형태의 행렬/벡터 기여분을 반환한다.
"""

import numpy as np
from scipy import sparse
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField, VectorField


class FVMSystem:
    """FVM 이산화 결과를 저장하는 선형 시스템."""

    def __init__(self, n: int):
        self.n = n
        # COO 형식 데이터
        self.rows: list = []
        self.cols: list = []
        self.vals: list = []
        self.rhs = np.zeros(n, dtype=np.float64)
        # 대각 계수 (SIMPLE 알고리즘에서 a_P 접근용)
        self.diag = np.zeros(n, dtype=np.float64)

    def add_diagonal(self, i: int, val: float):
        """대각 요소에 기여."""
        self.rows.append(i)
        self.cols.append(i)
        self.vals.append(val)
        self.diag[i] += val

    def add_off_diagonal(self, i: int, j: int, val: float):
        """비대각 요소에 기여."""
        self.rows.append(i)
        self.cols.append(j)
        self.vals.append(val)

    def add_source(self, i: int, val: float):
        """우변 벡터에 기여."""
        self.rhs[i] += val

    def to_sparse(self) -> sparse.csr_matrix:
        """CSR 희소 행렬 반환."""
        return sparse.coo_matrix(
            (self.vals, (self.rows, self.cols)),
            shape=(self.n, self.n)
        ).tocsr()

    def reset(self):
        """시스템 초기화."""
        self.rows.clear()
        self.cols.clear()
        self.vals.clear()
        self.rhs[:] = 0.0
        self.diag[:] = 0.0


def diffusion_operator(mesh: FVMesh, gamma: ScalarField, system: FVMSystem):
    """
    확산항 이산화: ∫∇·(Γ∇φ)dV ≈ Σ_f Γ_f·(φ_N - φ_P)/d_PN · A_f

    Parameters
    ----------
    mesh : FVMesh
    gamma : 확산 계수 필드 (셀 중심)
    system : 기여할 선형 시스템
    """
    for fid, face in enumerate(mesh.faces):
        owner = face.owner

        if face.neighbour >= 0:
            # 내부면
            neighbour = face.neighbour
            xO = mesh.cells[owner].center
            xN = mesh.cells[neighbour].center
            d_PN = np.linalg.norm(xN - xO)

            if d_PN < 1e-30:
                continue

            # 면에서의 확산 계수 (조화 평균)
            gamma_O = gamma.values[owner]
            gamma_N = gamma.values[neighbour]
            if gamma_O + gamma_N > 1e-30:
                gamma_f = 2.0 * gamma_O * gamma_N / (gamma_O + gamma_N)
            else:
                gamma_f = 0.0

            coeff = gamma_f * face.area / d_PN

            system.add_diagonal(owner, coeff)
            system.add_diagonal(neighbour, coeff)
            system.add_off_diagonal(owner, neighbour, -coeff)
            system.add_off_diagonal(neighbour, owner, -coeff)

        else:
            # 경계면: apply_boundary_conditions에서 BC 유형에 따라 처리
            # (Dirichlet → aP += D, b += D*phi_b; zero_gradient → 기여 없음)
            pass


def convection_operator_upwind(mesh: FVMesh, mass_flux: np.ndarray, system: FVMSystem):
    """
    대류항 이산화 (1차 Upwind): ∫∇·(ρuφ)dV ≈ Σ_f F_f · φ_f

    F_f = ρ·(u·n)·A (면 질량유속, 양수: owner→neighbour)

    Parameters
    ----------
    mass_flux : (n_faces,) 면 질량유속 배열
    """
    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        F = mass_flux[fid]

        if face.neighbour >= 0:
            neighbour = face.neighbour
            # Upwind: max(F,0)*φ_O + min(F,0)*φ_N
            system.add_diagonal(owner, max(F, 0.0))
            system.add_off_diagonal(owner, neighbour, min(F, 0.0))
            # 이웃셀: -F (반대 방향)
            system.add_diagonal(neighbour, max(-F, 0.0))
            system.add_off_diagonal(neighbour, owner, min(-F, 0.0))
        else:
            # 경계면
            if F >= 0:
                system.add_diagonal(owner, F)
            # F < 0: 경계값이 유입 → 우변에서 처리


def temporal_operator(mesh: FVMesh, rho: float, dt: float, phi_old: np.ndarray,
                      system: FVMSystem):
    """
    시간항 이산화 (Backward Euler):
    (ρ·V/Δt)·φ = (ρ·V/Δt)·φ° + ...

    Parameters
    ----------
    rho : 밀도
    dt : 시간 간격
    phi_old : 이전 시간 스텝의 필드 값
    """
    for ci in range(mesh.n_cells):
        vol = mesh.cells[ci].volume
        coeff = rho * vol / dt
        system.add_diagonal(ci, coeff)
        system.add_source(ci, coeff * phi_old[ci])


def source_term(mesh: FVMesh, source_values: np.ndarray, system: FVMSystem):
    """
    소스항: S_P·V_P를 우변에 추가.

    Parameters
    ----------
    source_values : (n_cells,) 체적당 소스항 값
    """
    for ci in range(mesh.n_cells):
        vol = mesh.cells[ci].volume
        system.add_source(ci, source_values[ci] * vol)


def linearized_source(mesh: FVMesh, Sp: np.ndarray, Su: np.ndarray,
                      system: FVMSystem):
    """
    선형화된 소스항: S = Su + Sp·φ

    Sp < 0이어야 안정적 (음의 기울기).
    대각에 -Sp·V, 우변에 Su·V 추가.
    """
    for ci in range(mesh.n_cells):
        vol = mesh.cells[ci].volume
        system.add_diagonal(ci, -Sp[ci] * vol)
        system.add_source(ci, Su[ci] * vol)


def apply_boundary_conditions(mesh: FVMesh, phi: ScalarField, gamma: ScalarField,
                              mass_flux: np.ndarray, system: FVMSystem,
                              bc_types: dict):
    """
    경계조건 적용.

    bc_types: {patch_name: {'type': 'dirichlet'|'neumann'|'zero_gradient', ...}}
    - dirichlet: 고정값 (phi.boundary_values에서 읽음)
    - neumann: 고정 기울기 (grad 값 지정)
    - zero_gradient: ∂φ/∂n = 0
    """
    for bname, fids in mesh.boundary_patches.items():
        bc = bc_types.get(bname, {'type': 'zero_gradient'})
        bc_type = bc['type']

        for local_idx, fid in enumerate(fids):
            face = mesh.faces[fid]
            owner = face.owner
            xO = mesh.cells[owner].center
            d_Pf = np.linalg.norm(face.center - xO)

            if d_Pf < 1e-30:
                continue

            if bc_type == 'dirichlet':
                phi_b = phi.boundary_values[bname][local_idx]
                gamma_f = gamma.values[owner]
                coeff = gamma_f * face.area / d_Pf
                system.add_diagonal(owner, coeff)
                system.add_source(owner, coeff * phi_b)

                # 대류 경계 (유입)
                F = mass_flux[fid]
                if F < 0:
                    system.add_source(owner, -F * phi_b)

            elif bc_type == 'neumann':
                grad_val = bc.get('value', 0.0)
                flux = grad_val * face.area
                system.add_source(owner, flux)

            elif bc_type == 'zero_gradient':
                # 확산: 기여 없음 (0-gradient)
                # 대류: 셀 값 사용
                F = mass_flux[fid]
                if F < 0:
                    # 유입인데 zero_gradient: 셀 값 사용
                    system.add_diagonal(owner, -F)


def under_relax(system: FVMSystem, phi: ScalarField, alpha: float):
    """
    하부완화 (Under-relaxation).

    a_P/α · φ = Σ(a_N·φ_N) + b + (1-α)/α · a_P · φ_old
    """
    if alpha >= 1.0:
        return

    for ci in range(system.n):
        ap = system.diag[ci]
        system.add_diagonal(ci, ap * (1.0 - alpha) / alpha)
        system.add_source(ci, ap * (1.0 - alpha) / alpha * phi.values[ci])
