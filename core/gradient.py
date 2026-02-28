"""
셀 중심 기울기 복원.

Green-Gauss 방법과 Least Squares 방법을 제공한다.
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField


def green_gauss_gradient(phi: ScalarField) -> np.ndarray:
    """
    Green-Gauss 기울기 복원.

    ∇φ_P ≈ (1/V_P) Σ_f φ_f · n_f · A_f

    Parameters
    ----------
    phi : 스칼라 필드

    Returns
    -------
    grad : (n_cells, 2) 기울기 배열
    """
    mesh = phi.mesh
    ndim = getattr(mesh, 'ndim', 2)
    grad = np.zeros((mesh.n_cells, ndim), dtype=np.float64)

    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        if face.neighbour >= 0:
            gc = _interpolation_weight(mesh, fid)
            phi_f = gc * phi.values[owner] + (1.0 - gc) * phi.values[face.neighbour]
        else:
            phi_f = _get_boundary_face_value(phi, fid)

        flux = phi_f * face.normal[:ndim] * face.area
        grad[owner] += flux
        if face.neighbour >= 0:
            grad[face.neighbour] -= flux

    for ci in range(mesh.n_cells):
        vol = mesh.cells[ci].volume
        if vol > 1e-30:
            grad[ci] /= vol

    return grad


def least_squares_gradient(phi: ScalarField) -> np.ndarray:
    """
    Least Squares 기울기 복원.

    이웃 셀과의 차이를 사용하여 기울기를 최소자승법으로 계산.

    Returns
    -------
    grad : (n_cells, 2) 기울기 배열
    """
    mesh = phi.mesh
    ndim = getattr(mesh, 'ndim', 2)
    grad = np.zeros((mesh.n_cells, ndim), dtype=np.float64)

    for ci in range(mesh.n_cells):
        cell = mesh.cells[ci]
        xP = cell.center[:ndim]

        dxs = []
        dphis = []

        for fid in cell.faces:
            face = mesh.faces[fid]
            if face.neighbour >= 0:
                nb = face.neighbour if face.owner == ci else face.owner
                xN = mesh.cells[nb].center[:ndim]
                dx = xN - xP
                dphi = phi.values[nb] - phi.values[ci]
            else:
                dx = face.center[:ndim] - xP
                phi_f = _get_boundary_face_value(phi, fid)
                dphi = phi_f - phi.values[ci]
            dxs.append(dx)
            dphis.append(dphi)

        if len(dxs) < ndim:
            continue

        A = np.array(dxs)  # (n_neighbours, ndim)
        b = np.array(dphis)  # (n_neighbours,)

        ATA = A.T @ A  # (ndim, ndim)
        ATb = A.T @ b  # (ndim,)
        try:
            grad[ci] = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            pass  # singular matrix, leave as zero

    return grad


def _interpolation_weight(mesh: FVMesh, face_id: int) -> float:
    """내부면의 보간 가중치 (소유셀 기준) 계산."""
    face = mesh.faces[face_id]
    xO = mesh.cells[face.owner].center
    xN = mesh.cells[face.neighbour].center
    xF = face.center

    dO = np.linalg.norm(xF - xO)
    dN = np.linalg.norm(xF - xN)
    total = dO + dN
    if total < 1e-30:
        return 0.5
    return dN / total  # 소유셀에 가까울수록 가중치 큼


def _get_boundary_face_value(phi: ScalarField, face_id: int) -> float:
    """경계면에서의 스칼라 값."""
    mesh = phi.mesh
    face = mesh.faces[face_id]
    for bname, fids in mesh.boundary_patches.items():
        try:
            local_idx = fids.index(face_id)
            return phi.boundary_values[bname][local_idx]
        except ValueError:
            continue
    return phi.values[face.owner]
