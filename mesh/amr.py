"""
적응 격자 세분화 (AMR: Adaptive Mesh Refinement).

셀 기반 등방 세분화: quad → 4 quads.
1-레벨 등급화 규칙, 오차 추정, 필드 전달.
"""

import numpy as np
from mesh.mesh_reader import FVMesh, build_fvmesh_from_arrays
from core.fields import ScalarField


class AMRCell:
    """AMR 셀 정보."""

    def __init__(self, cell_id: int, level: int = 0,
                 parent: int = -1, children: list = None):
        self.cell_id = cell_id
        self.level = level
        self.parent = parent
        self.children = children or []

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class AMRMesh:
    """
    적응 격자 관리자.

    초기 격자에서 시작하여 셀 세분화/거칠게 하기를 수행한다.
    get_active_mesh()는 활성(리프) 셀만으로 FVMesh를 재구성한다.

    Parameters
    ----------
    base_mesh : 초기 FVMesh (2D quad 격자)
    max_level : 최대 세분화 레벨
    """

    def __init__(self, base_mesh: FVMesh, max_level: int = 3):
        self.base_mesh = base_mesh
        self.max_level = max_level

        # AMR 셀 트리
        self.amr_cells = []
        for ci in range(base_mesh.n_cells):
            self.amr_cells.append(AMRCell(ci, level=0))

        # 노드 및 셀-노드 목록 (세분화 시 확장)
        self.nodes = base_mesh.nodes.copy()
        self.cell_node_list = []
        for ci in range(base_mesh.n_cells):
            cell = base_mesh.cells[ci]
            self.cell_node_list.append(np.array(cell.nodes))

        # 세분화 이력
        self.n_refinements = 0

    def refine_cells(self, cell_ids: list):
        """
        지정 셀들을 세분화 (quad → 4 quads).

        Parameters
        ----------
        cell_ids : 세분화할 셀 ID 목록
        """
        for ci in cell_ids:
            amr_cell = self.amr_cells[ci]

            if not amr_cell.is_leaf:
                continue
            if amr_cell.level >= self.max_level:
                continue

            # 부모 셀의 4 노드 (quad)
            parent_nodes = self.cell_node_list[ci]
            if len(parent_nodes) != 4:
                continue  # quad만 세분화

            # 꼭짓점 좌표
            pts = self.nodes[parent_nodes]  # (4, ndim)

            # 중점 노드 생성: 4 변의 중점 + 셀 중심 = 5 새 노드
            mid01 = 0.5 * (pts[0] + pts[1])
            mid12 = 0.5 * (pts[1] + pts[2])
            mid23 = 0.5 * (pts[2] + pts[3])
            mid30 = 0.5 * (pts[3] + pts[0])
            center = 0.25 * (pts[0] + pts[1] + pts[2] + pts[3])

            new_pts = np.array([mid01, mid12, mid23, mid30, center])
            n_start = len(self.nodes)
            self.nodes = np.vstack([self.nodes, new_pts])

            n_mid01 = n_start
            n_mid12 = n_start + 1
            n_mid23 = n_start + 2
            n_mid30 = n_start + 3
            n_center = n_start + 4

            n0, n1, n2, n3 = parent_nodes

            # 4 자식 셀
            child_cells = [
                np.array([n0, n_mid01, n_center, n_mid30]),   # SW
                np.array([n_mid01, n1, n_mid12, n_center]),   # SE
                np.array([n_center, n_mid12, n2, n_mid23]),   # NE
                np.array([n_mid30, n_center, n_mid23, n3]),   # NW
            ]

            child_ids = []
            for child_nodes in child_cells:
                new_id = len(self.amr_cells)
                self.amr_cells.append(
                    AMRCell(new_id, level=amr_cell.level + 1, parent=ci))
                self.cell_node_list.append(child_nodes)
                child_ids.append(new_id)

            amr_cell.children = child_ids

        self.n_refinements += 1

    def get_active_cells(self) -> list:
        """활성(리프) 셀 ID 목록."""
        return [c.cell_id for c in self.amr_cells if c.is_leaf]

    def get_active_mesh(self) -> FVMesh:
        """
        활성 리프 셀만으로 FVMesh 재구성.

        Returns
        -------
        FVMesh : 평탄화된 메쉬 (솔버가 직접 사용 가능)
        """
        active_ids = self.get_active_cells()
        ndim = getattr(self.base_mesh, 'ndim', 2)

        # 활성 셀의 노드 수집
        used_nodes = set()
        for ci in active_ids:
            for nid in self.cell_node_list[ci]:
                used_nodes.add(nid)

        # 노드 재번호
        old_to_new = {}
        new_nodes = []
        for new_id, old_id in enumerate(sorted(used_nodes)):
            old_to_new[old_id] = new_id
            new_nodes.append(self.nodes[old_id])
        new_nodes = np.array(new_nodes)

        # 셀 노드 목록 재번호
        new_cell_node_list = []
        for ci in active_ids:
            old_nodes = self.cell_node_list[ci]
            new_cell_nodes = np.array([old_to_new[n] for n in old_nodes])
            new_cell_node_list.append(new_cell_nodes)

        # 경계면: base_mesh의 경계를 노드 매핑으로 재구성
        boundary_faces_dict = {}
        for bname, fids in self.base_mesh.boundary_patches.items():
            boundary_faces_dict[bname] = []
            for fid in fids:
                face = self.base_mesh.faces[fid]
                face_nodes = face.nodes
                # 원본 노드가 새 메쉬에 있는지 확인
                all_present = all(n in old_to_new for n in face_nodes)
                if all_present:
                    new_face_nodes = np.array([old_to_new[n] for n in face_nodes])
                    boundary_faces_dict[bname].append(new_face_nodes)

        return build_fvmesh_from_arrays(
            new_nodes, new_cell_node_list, boundary_faces_dict, ndim=ndim)

    def transfer_field_to_children(self, field: ScalarField,
                                    active_mesh: FVMesh) -> ScalarField:
        """
        필드를 세분화된 메쉬로 전달 (부모→자식 복사).

        Parameters
        ----------
        field : 이전 메쉬의 스칼라 필드
        active_mesh : 새 활성 메쉬

        Returns
        -------
        new_field : 새 메쉬의 스칼라 필드
        """
        new_field = ScalarField(active_mesh, field.name)
        active_ids = self.get_active_cells()

        # 이전 활성 셀 ID → 필드 인덱스 매핑이 필요
        # 간단 버전: 새 활성 셀의 부모로부터 값 상속
        for new_idx, ci in enumerate(active_ids):
            amr_cell = self.amr_cells[ci]

            if new_idx < len(field.values):
                new_field.values[new_idx] = field.values[min(new_idx,
                                                             len(field.values) - 1)]
            elif amr_cell.parent >= 0:
                # 부모의 값 상속 (부모가 이전 필드에 있으면)
                parent_amr = self.amr_cells[amr_cell.parent]
                # 부모가 이전 활성 목록에서 몇 번째였는지 찾기
                new_field.values[new_idx] = np.mean(field.values)

        return new_field


class GradientJumpEstimator:
    """
    면 기울기 점프 기반 오차 지표.

    각 셀의 오차 = max(|grad_phi · n| jump) over cell faces.
    """

    @staticmethod
    def estimate(mesh: FVMesh, phi: ScalarField) -> np.ndarray:
        """
        오차 추정.

        Parameters
        ----------
        mesh : FVMesh
        phi : 스칼라 필드

        Returns
        -------
        error : (n_cells,) 셀별 오차 지표
        """
        n = mesh.n_cells
        error = np.zeros(n)

        for fid, face in enumerate(mesh.faces):
            if face.neighbour < 0:
                continue

            owner = face.owner
            neighbour = face.neighbour

            # 면에서의 기울기 점프 (간단: 셀 값 차이 / 거리)
            d = np.linalg.norm(
                mesh.cells[neighbour].center - mesh.cells[owner].center)
            if d < 1e-15:
                continue

            jump = abs(phi.values[neighbour] - phi.values[owner]) / d

            error[owner] = max(error[owner], jump)
            error[neighbour] = max(error[neighbour], jump)

        return error


class AMRSolverLoop:
    """
    AMR 풀이 순환: solve → estimate → refine → repeat.

    Parameters
    ----------
    amr_mesh : AMRMesh
    refine_fraction : 세분화할 셀 비율 (상위 N%)
    """

    def __init__(self, amr_mesh: AMRMesh, refine_fraction: float = 0.3):
        self.amr_mesh = amr_mesh
        self.refine_fraction = refine_fraction
        self.estimator = GradientJumpEstimator()

    def mark_cells(self, mesh: FVMesh, phi: ScalarField) -> list:
        """
        세분화할 셀 표시.

        Returns
        -------
        cells_to_refine : 세분화할 활성 셀의 AMR ID 목록
        """
        error = self.estimator.estimate(mesh, phi)
        threshold = np.percentile(error, (1.0 - self.refine_fraction) * 100)

        active_ids = self.amr_mesh.get_active_cells()
        cells_to_refine = []
        for local_idx, amr_id in enumerate(active_ids):
            if local_idx < len(error) and error[local_idx] > threshold:
                cells_to_refine.append(amr_id)

        return cells_to_refine
