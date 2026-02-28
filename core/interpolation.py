"""
면 보간 스킴: Upwind, Central Differencing, TVD/MUSCL.

대류항 이산화에서 면 값을 결정하는 데 사용된다.
제한자 레지스트리와 MUSCL 재구성 (지연 보정 패턴) 포함.
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField


# =====================================================================
# TVD 제한자 레지스트리
# =====================================================================

def limiter_minmod(r: np.ndarray) -> np.ndarray:
    """Minmod 제한자: max(0, min(1, r))."""
    return np.maximum(0.0, np.minimum(1.0, r))


def limiter_superbee(r: np.ndarray) -> np.ndarray:
    """Superbee 제한자: max(0, min(2r,1), min(r,2))."""
    return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0),
                                       np.minimum(r, 2.0)))


def limiter_van_albada(r: np.ndarray) -> np.ndarray:
    """Van Albada 제한자: (r^2 + r) / (r^2 + 1)."""
    return np.where(r > 0, (r * r + r) / (r * r + 1.0), 0.0)


def limiter_van_leer(r: np.ndarray) -> np.ndarray:
    """Van Leer 제한자: (r + |r|) / (1 + |r|)."""
    return np.where(r > 0, 2.0 * r / (1.0 + r), 0.0)


# 제한자 레지스트리
LIMITER_REGISTRY = {
    'minmod': limiter_minmod,
    'superbee': limiter_superbee,
    'van_albada': limiter_van_albada,
    'van_leer': limiter_van_leer,
}


def get_limiter(name: str):
    """이름으로 제한자 함수를 가져온다."""
    if name not in LIMITER_REGISTRY:
        raise ValueError(f"Unknown limiter: {name}. "
                         f"Available: {list(LIMITER_REGISTRY.keys())}")
    return LIMITER_REGISTRY[name]


def upwind_face_value(phi: ScalarField, face_id: int, flux_dir: float) -> float:
    """
    1차 Upwind 보간.

    flux_dir > 0이면 소유셀 값, < 0이면 이웃셀 값 사용.
    """
    face = phi.mesh.faces[face_id]
    if face.neighbour < 0:
        return _get_boundary_value(phi, face_id)

    if flux_dir >= 0:
        return phi.values[face.owner]
    else:
        return phi.values[face.neighbour]


def central_face_value(phi: ScalarField, face_id: int) -> float:
    """
    중심차분 (Central Differencing) 보간.

    면 값 = gc * φ_O + (1-gc) * φ_N
    """
    face = phi.mesh.faces[face_id]
    if face.neighbour < 0:
        return _get_boundary_value(phi, face_id)

    gc = _interp_weight(phi.mesh, face_id)
    return gc * phi.values[face.owner] + (1.0 - gc) * phi.values[face.neighbour]


def tvd_face_value(phi: ScalarField, face_id: int, flux_dir: float,
                   grad_phi: np.ndarray) -> float:
    """
    TVD 보간 (Van Leer limiter).

    2차 정확도를 유지하면서 진동을 억제한다.
    """
    face = phi.mesh.faces[face_id]
    mesh = phi.mesh

    if face.neighbour < 0:
        return _get_boundary_value(phi, face_id)

    owner = face.owner
    neighbour = face.neighbour

    if flux_dir >= 0:
        upwind_cell = owner
        downwind_cell = neighbour
    else:
        upwind_cell = neighbour
        downwind_cell = owner

    phi_U = phi.values[upwind_cell]
    phi_D = phi.values[downwind_cell]

    # 기울기를 사용하여 upwind 셀에서의 면 값 추정
    dx = face.center - mesh.cells[upwind_cell].center
    phi_face_linear = phi_U + np.dot(grad_phi[upwind_cell], dx)

    # r = (φ_D - φ_U) / (2*(φ_face_linear - φ_U)) 비율 계산
    delta_upwind = phi_face_linear - phi_U
    delta_total = phi_D - phi_U

    if abs(delta_total) < 1e-15:
        return phi_U

    r = 2.0 * delta_upwind / delta_total - 1.0 if abs(delta_total) > 1e-15 else 0.0

    # Van Leer limiter: ψ(r) = (r + |r|) / (1 + |r|)
    psi = (r + abs(r)) / (1.0 + abs(r)) if (1.0 + abs(r)) > 1e-15 else 0.0

    # 면 값 = φ_U + ψ/2 * (φ_D - φ_U)
    return phi_U + 0.5 * psi * delta_total


def compute_face_values_upwind(phi: ScalarField, mass_flux: np.ndarray) -> np.ndarray:
    """
    모든 면에 대해 Upwind 보간 적용.

    Parameters
    ----------
    phi : 스칼라 필드
    mass_flux : (n_faces,) 각 면의 질량유속 (양: 소유셀→이웃셀)

    Returns
    -------
    phi_f : (n_faces,) 면 값 배열
    """
    mesh = phi.mesh
    n_faces = mesh.n_faces
    phi_f = np.zeros(n_faces)

    for fid in range(n_faces):
        phi_f[fid] = upwind_face_value(phi, fid, mass_flux[fid])

    return phi_f


def compute_face_values_central(phi: ScalarField) -> np.ndarray:
    """모든 면에 대해 Central 보간 적용."""
    mesh = phi.mesh
    phi_f = np.zeros(mesh.n_faces)
    for fid in range(mesh.n_faces):
        phi_f[fid] = central_face_value(phi, fid)
    return phi_f


def compute_mass_flux(U: 'VectorField', rho: float, mesh: FVMesh) -> np.ndarray:
    """
    속도장과 밀도로부터 면 질량유속 계산.

    F_f = ρ · (u_f · n_f) · A_f
    """
    n_faces = mesh.n_faces
    mass_flux = np.zeros(n_faces)

    for fid, face in enumerate(mesh.faces):
        if face.neighbour >= 0:
            gc = _interp_weight(mesh, fid)
            u_f = gc * U.values[face.owner] + (1.0 - gc) * U.values[face.neighbour]
        else:
            # 경계
            u_f = _get_boundary_vector_value(U, fid)

        mass_flux[fid] = rho * np.dot(u_f, face.normal) * face.area

    return mass_flux


def _interp_weight(mesh: FVMesh, face_id: int) -> float:
    """소유셀 가중치."""
    face = mesh.faces[face_id]
    xO = mesh.cells[face.owner].center
    xN = mesh.cells[face.neighbour].center
    xF = face.center
    dO = np.linalg.norm(xF - xO)
    dN = np.linalg.norm(xF - xN)
    total = dO + dN
    if total < 1e-30:
        return 0.5
    return dN / total


def _get_boundary_value(phi: ScalarField, face_id: int) -> float:
    """경계면 스칼라 값."""
    mesh = phi.mesh
    for bname, fids in mesh.boundary_patches.items():
        try:
            local_idx = fids.index(face_id)
            return phi.boundary_values[bname][local_idx]
        except ValueError:
            continue
    return phi.values[mesh.faces[face_id].owner]


def _get_boundary_vector_value(U, face_id: int) -> np.ndarray:
    """경계면 벡터 값."""
    mesh = U.mesh
    for bname, fids in mesh.boundary_patches.items():
        try:
            local_idx = fids.index(face_id)
            return U.boundary_values[bname][local_idx]
        except ValueError:
            continue
    return U.values[mesh.faces[face_id].owner]


# =====================================================================
# MUSCL 지연 보정 (Deferred Correction)
# =====================================================================

def muscl_deferred_correction(mesh: FVMesh, phi: ScalarField,
                               mass_flux: np.ndarray,
                               grad_phi: np.ndarray,
                               limiter_name: str = 'van_leer') -> np.ndarray:
    """
    MUSCL 지연 보정 소스항 계산.

    지연 보정 패턴:
      φ_f^{MUSCL} = φ_f^{UW} + ψ(r) · (φ_f^{HO} - φ_f^{UW})

    암묵적 부분은 Upwind (행렬에 반영), 고차 보정은 RHS에 추가.
    이 함수는 각 셀의 RHS 보정항을 반환한다.

    Parameters
    ----------
    mesh : FVMesh
    phi : 스칼라 필드
    mass_flux : 면 질량유속
    grad_phi : (n_cells, ndim) 셀 중심 기울기
    limiter_name : 제한자 이름 ('minmod', 'superbee', 'van_albada', 'van_leer')

    Returns
    -------
    correction : (n_cells,) 각 셀의 RHS 보정항
    """
    limiter_func = get_limiter(limiter_name)
    n_cells = mesh.n_cells
    correction = np.zeros(n_cells)

    for fid, face in enumerate(mesh.faces):
        if face.neighbour < 0:
            continue  # 경계면은 보정 없음

        owner = face.owner
        neighbour = face.neighbour
        F = mass_flux[fid]

        if F >= 0:
            upwind_cell = owner
            downwind_cell = neighbour
        else:
            upwind_cell = neighbour
            downwind_cell = owner

        phi_U = phi.values[upwind_cell]
        phi_D = phi.values[downwind_cell]

        # upwind 셀 기울기로 면 값 추정 (선형 재구성)
        dx = face.center - mesh.cells[upwind_cell].center
        phi_face_linear = phi_U + np.dot(grad_phi[upwind_cell], dx)

        # r 비율: upwind 기울기 대 셀 간 기울기
        delta_face = phi_face_linear - phi_U
        delta_CD = phi_D - phi_U

        if abs(delta_CD) < 1e-15:
            continue

        r = 2.0 * delta_face / delta_CD - 1.0

        # 제한자 적용
        psi = float(limiter_func(np.array([r]))[0])

        # 고차 보정: F * (φ_MUSCL - φ_UW)
        phi_UW = phi_U  # upwind 값
        phi_MUSCL = phi_U + 0.5 * psi * delta_CD
        flux_correction = abs(F) * (phi_MUSCL - phi_UW)

        # 소유셀/이웃셀 기여 (보정은 면 플럭스의 차이)
        if F >= 0:
            correction[owner] -= flux_correction
            correction[neighbour] += flux_correction
        else:
            correction[neighbour] -= flux_correction
            correction[owner] += flux_correction

    return correction
