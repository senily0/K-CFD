"""
RCB(재귀 좌표 이분할) 기반 격자 분할 및 고스트 셀.

교육용 코드에 적합한 간단한 기하 분할 방법.
"""

import numpy as np
from mesh.mesh_reader import FVMesh


class GeometricPartitioner:
    """
    재귀 좌표 이분할 (Recursive Coordinate Bisection).

    셀 중심 좌표를 기준으로 도메인을 분할한다.
    """

    def __init__(self, mesh: FVMesh):
        self.mesh = mesh

    def partition(self, n_parts: int) -> np.ndarray:
        """
        격자를 n_parts 파트로 분할.

        Parameters
        ----------
        n_parts : 분할 수 (2의 거듭제곱 권장)

        Returns
        -------
        part_ids : (n_cells,) 각 셀의 파티션 번호 (0..n_parts-1)
        """
        mesh = self.mesh
        n = mesh.n_cells
        ndim = getattr(mesh, 'ndim', 2)

        # 셀 중심 좌표
        centers = np.array([mesh.cells[ci].center[:ndim] for ci in range(n)])

        # 재귀 이분할
        part_ids = np.zeros(n, dtype=int)
        self._rcb_split(centers, np.arange(n), part_ids, 0, n_parts, 0, ndim)

        return part_ids

    def _rcb_split(self, centers, cell_ids, part_ids,
                   current_part, n_parts, depth, ndim):
        """재귀 좌표 이분할."""
        if n_parts <= 1 or len(cell_ids) == 0:
            part_ids[cell_ids] = current_part
            return

        # 분할 축: depth % ndim
        axis = depth % ndim
        coords = centers[cell_ids, axis]

        # 중간값으로 이분
        median = np.median(coords)
        left_mask = coords <= median
        right_mask = ~left_mask

        # 최소 한 셀은 각 쪽에
        if not np.any(left_mask):
            left_mask[0] = True
            right_mask[0] = False
        if not np.any(right_mask):
            right_mask[-1] = True
            left_mask[-1] = False

        left_ids = cell_ids[left_mask]
        right_ids = cell_ids[right_mask]

        n_left = n_parts // 2
        n_right = n_parts - n_left

        self._rcb_split(centers, left_ids, part_ids,
                        current_part, n_left, depth + 1, ndim)
        self._rcb_split(centers, right_ids, part_ids,
                        current_part + n_left, n_right, depth + 1, ndim)


class GhostCellLayer:
    """
    고스트 셀 레이어 구성.

    파티션 경계면의 이웃 셀을 고스트로 추가한다.
    """

    def __init__(self, mesh: FVMesh, part_ids: np.ndarray):
        self.mesh = mesh
        self.part_ids = part_ids
        self.n_parts = int(np.max(part_ids)) + 1

        # 각 파티션의 고스트 셀 맵
        # ghost_map[rank] = {remote_cell_id: local_ghost_idx, ...}
        self.ghost_map = {}
        # send_map[rank] = [(to_rank, cell_ids), ...]
        self.send_map = {}
        self.recv_map = {}

        self._build_ghost_maps()

    def _build_ghost_maps(self):
        """파티션 경계면에서 고스트 셀 맵 구성."""
        mesh = self.mesh
        n_parts = self.n_parts

        for rank in range(n_parts):
            self.ghost_map[rank] = {}
            self.send_map[rank] = {}
            self.recv_map[rank] = {}

        for fid, face in enumerate(mesh.faces):
            if face.neighbour < 0:
                continue

            o_part = self.part_ids[face.owner]
            n_part = self.part_ids[face.neighbour]

            if o_part != n_part:
                # owner 파티션에서 neighbour는 고스트
                if n_part not in self.ghost_map[o_part]:
                    self.ghost_map[o_part][n_part] = set()
                self.ghost_map[o_part][n_part].add(face.neighbour)

                # neighbour 파티션에서 owner는 고스트
                if o_part not in self.ghost_map[n_part]:
                    self.ghost_map[n_part][o_part] = set()
                self.ghost_map[n_part][o_part].add(face.owner)

        # set → sorted list
        for rank in range(n_parts):
            for remote_rank in list(self.ghost_map[rank].keys()):
                self.ghost_map[rank][remote_rank] = sorted(
                    self.ghost_map[rank][remote_rank])

    def get_local_cells(self, rank: int) -> np.ndarray:
        """파티션 rank의 로컬 셀 ID 목록."""
        return np.where(self.part_ids == rank)[0]

    def get_ghost_cells(self, rank: int) -> dict:
        """파티션 rank의 고스트 셀: {remote_rank: [cell_ids]}."""
        return self.ghost_map.get(rank, {})
