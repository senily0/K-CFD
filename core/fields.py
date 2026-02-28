"""
셀 중심 유한체적법을 위한 스칼라/벡터 필드 클래스.

각 필드는 메쉬의 셀 중심에서 정의되며, 경계값 저장 기능을 포함한다.
"""

import numpy as np
from typing import Dict, Optional
from mesh.mesh_reader import FVMesh


class ScalarField:
    """셀 중심 스칼라 필드."""

    def __init__(self, mesh: FVMesh, name: str = "scalar", default: float = 0.0):
        self.mesh = mesh
        self.name = name
        self.values = np.full(mesh.n_cells, default, dtype=np.float64)
        self.boundary_values: Dict[str, np.ndarray] = {}
        # 각 경계 패치별 면 값 초기화
        for bname, fids in mesh.boundary_patches.items():
            self.boundary_values[bname] = np.full(len(fids), default, dtype=np.float64)
        # 이전 시간 스텝 값 (비정상 계산용)
        self.old_values: Optional[np.ndarray] = None

    def copy(self) -> 'ScalarField':
        """필드 복사."""
        sf = ScalarField(self.mesh, self.name)
        sf.values = self.values.copy()
        for bname in self.boundary_values:
            sf.boundary_values[bname] = self.boundary_values[bname].copy()
        if self.old_values is not None:
            sf.old_values = self.old_values.copy()
        return sf

    def store_old(self):
        """현재 값을 이전 시간 스텝으로 저장."""
        self.old_values = self.values.copy()

    def set_uniform(self, value: float):
        """전체 필드를 균일값으로 설정."""
        self.values[:] = value
        for bname in self.boundary_values:
            self.boundary_values[bname][:] = value

    def set_boundary(self, patch_name: str, value):
        """경계 패치에 값 설정 (스칼라 또는 배열)."""
        if patch_name not in self.boundary_values:
            return
        if np.isscalar(value):
            self.boundary_values[patch_name][:] = value
        else:
            self.boundary_values[patch_name] = np.array(value, dtype=np.float64)

    def get_face_value(self, face_idx: int) -> float:
        """특정 면에서의 값 (경계면이면 경계값, 내부면이면 소유셀 값)."""
        face = self.mesh.faces[face_idx]
        if face.neighbour == -1:
            # 경계면 → 해당 패치에서 찾기
            for bname, fids in self.mesh.boundary_patches.items():
                if face_idx in fids:
                    local_idx = fids.index(face_idx)
                    return self.boundary_values[bname][local_idx]
            return self.values[face.owner]
        return self.values[face.owner]

    def max(self) -> float:
        return float(np.max(self.values))

    def min(self) -> float:
        return float(np.min(self.values))

    def mean(self) -> float:
        return float(np.mean(self.values))


class VectorField:
    """셀 중심 벡터 필드 (2D/3D, mesh.ndim 기반)."""

    def __init__(self, mesh: FVMesh, name: str = "vector", default: np.ndarray = None):
        self.mesh = mesh
        self.name = name
        ndim = getattr(mesh, 'ndim', 2)
        if default is None:
            default = np.zeros(ndim)
        else:
            default = np.array(default, dtype=np.float64)
            if len(default) < ndim:
                default = np.pad(default, (0, ndim - len(default)))
        self.values = np.tile(default, (mesh.n_cells, 1)).astype(np.float64)
        self.boundary_values: Dict[str, np.ndarray] = {}
        for bname, fids in mesh.boundary_patches.items():
            self.boundary_values[bname] = np.tile(default, (len(fids), 1)).astype(np.float64)
        self.old_values: Optional[np.ndarray] = None

    @property
    def x(self) -> np.ndarray:
        """x-성분 배열."""
        return self.values[:, 0]

    @x.setter
    def x(self, val):
        self.values[:, 0] = val

    @property
    def y(self) -> np.ndarray:
        """y-성분 배열."""
        return self.values[:, 1]

    @y.setter
    def y(self, val):
        self.values[:, 1] = val

    @property
    def z(self) -> np.ndarray:
        """z-성분 배열 (3D 전용)."""
        if self.values.shape[1] >= 3:
            return self.values[:, 2]
        return np.zeros(self.values.shape[0])

    @z.setter
    def z(self, val):
        if self.values.shape[1] >= 3:
            self.values[:, 2] = val

    def copy(self) -> 'VectorField':
        vf = VectorField(self.mesh, self.name)
        vf.values = self.values.copy()
        for bname in self.boundary_values:
            vf.boundary_values[bname] = self.boundary_values[bname].copy()
        if self.old_values is not None:
            vf.old_values = self.old_values.copy()
        return vf

    def store_old(self):
        self.old_values = self.values.copy()

    def set_uniform(self, value: np.ndarray):
        value = np.array(value, dtype=np.float64)
        self.values[:] = value
        for bname in self.boundary_values:
            self.boundary_values[bname][:] = value

    def set_boundary(self, patch_name: str, value):
        if patch_name not in self.boundary_values:
            return
        value = np.array(value, dtype=np.float64)
        if value.ndim == 1:
            self.boundary_values[patch_name][:] = value
        else:
            self.boundary_values[patch_name] = value

    def magnitude(self) -> np.ndarray:
        """벡터 크기 배열 (ndim 무관)."""
        return np.sqrt(np.sum(self.values**2, axis=1))
