"""
Gmsh .msh 파일 파서 및 내부 메쉬 데이터 구조.

FVM 해석을 위한 셀-면-노드 연결 정보, 면적/체적 계산,
경계면 식별 등을 제공한다.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Face:
    """메쉬 면(face) 데이터."""
    nodes: np.ndarray          # 면을 구성하는 노드 인덱스
    owner: int = -1            # 소유 셀 인덱스
    neighbour: int = -1        # 이웃 셀 인덱스 (-1이면 경계면)
    area: float = 0.0          # 면적
    normal: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 외향 법선벡터
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 면 중심 좌표
    boundary_tag: str = ""     # 경계 태그 (내부면이면 "")


@dataclass
class Cell:
    """메쉬 셀(cell) 데이터."""
    nodes: np.ndarray          # 셀을 구성하는 노드 인덱스
    faces: List[int] = field(default_factory=list)  # 셀에 속하는 면 인덱스
    volume: float = 0.0        # 셀 체적 (2D: 면적)
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 셀 중심 좌표


class FVMesh:
    """
    FVM 해석을 위한 2D/3D 비정렬 메쉬 데이터 구조.

    셀 중심 유한체적법에 필요한 모든 기하학적 정보를 저장한다.
    ndim 속성이 공간 차원의 단일 진실 소스이다.
    """

    def __init__(self, ndim: int = 2):
        self.ndim: int = ndim
        self.nodes: np.ndarray = np.empty((0, ndim))
        self.cells: List[Cell] = []
        self.faces: List[Face] = []
        self.n_cells: int = 0
        self.n_faces: int = 0
        self.n_internal_faces: int = 0
        self.n_boundary_faces: int = 0
        self.boundary_patches: Dict[str, List[int]] = {}
        self.cell_zones: Dict[str, List[int]] = {}

    def summary(self) -> str:
        """메쉬 요약 정보."""
        s = f"FVMesh({self.ndim}D): {self.n_cells} cells, {self.n_faces} faces "
        s += f"({self.n_internal_faces} internal, {self.n_boundary_faces} boundary)\n"
        s += f"  Nodes: {len(self.nodes)}\n"
        for name, fids in self.boundary_patches.items():
            s += f"  Boundary '{name}': {len(fids)} faces\n"
        for name, cids in self.cell_zones.items():
            s += f"  Zone '{name}': {len(cids)} cells\n"
        return s


def _compute_face_geometry_2d(nodes: np.ndarray, face_node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """2D 면(에지)의 중심, 법선, 길이 계산."""
    p0 = nodes[face_node_ids[0]]
    p1 = nodes[face_node_ids[1]]
    center = 0.5 * (p0 + p1)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = np.sqrt(dx**2 + dy**2)
    # 외향 법선 (반시계 방향 기준 오른쪽)
    normal = np.array([dy, -dx])
    if length > 1e-15:
        normal /= length
    return center, normal, length


def _compute_face_geometry_3d(nodes: np.ndarray, face_node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """3D 다각형 면의 중심, 법선, 면적 계산 (Newell 방법)."""
    pts = nodes[face_node_ids]
    n = len(pts)
    center = np.mean(pts, axis=0)

    # Newell's method for polygon normal
    normal = np.zeros(3)
    for i in range(n):
        j = (i + 1) % n
        normal[0] += (pts[i, 1] - pts[j, 1]) * (pts[i, 2] + pts[j, 2])
        normal[1] += (pts[i, 2] - pts[j, 2]) * (pts[i, 0] + pts[j, 0])
        normal[2] += (pts[i, 0] - pts[j, 0]) * (pts[i, 1] + pts[j, 1])

    area = np.linalg.norm(normal)
    if area > 1e-15:
        normal /= area
    area *= 0.5
    return center, normal, area


def _compute_cell_geometry_2d(nodes: np.ndarray, cell_node_ids: np.ndarray) -> Tuple[np.ndarray, float]:
    """2D 셀의 중심 및 면적 계산 (shoelace 공식)."""
    pts = nodes[cell_node_ids]
    n = len(pts)
    center = np.mean(pts, axis=0)
    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
    area = abs(area) / 2.0
    return center, area


def _compute_cell_geometry_3d(nodes: np.ndarray, cell_node_ids: np.ndarray,
                               cell_face_ids: List[int], faces: List[Face]) -> Tuple[np.ndarray, float]:
    """3D 셀의 중심 및 체적 계산 (발산정리 기반)."""
    pts = nodes[cell_node_ids]
    center = np.mean(pts, axis=0)

    # 발산정리: V = (1/3) Σ_f (x_f · n_f) * A_f
    volume = 0.0
    for fid in cell_face_ids:
        face = faces[fid]
        # 법선이 셀 바깥을 향하는지 확인
        sign = 1.0
        if face.owner != -1:
            to_face = face.center - center
            if np.dot(face.normal, to_face) < 0:
                sign = -1.0
        volume += sign * np.dot(face.center, face.normal) * face.area

    volume = abs(volume) / 3.0
    if volume < 1e-30:
        # 폴백: 볼록 다면체 근사
        volume = 1e-30
    return center, volume


def build_fvmesh_from_arrays(
    node_coords: np.ndarray,
    cell_node_list: List[np.ndarray],
    boundary_faces_dict: Dict[str, List[np.ndarray]],
    cell_zone_map: Optional[Dict[str, List[int]]] = None,
    ndim: int = 2,
    cell_face_node_list: Optional[List[List[np.ndarray]]] = None
) -> FVMesh:
    """
    노드 좌표, 셀-노드 연결, 경계면 정보로부터 FVMesh 구성.

    Parameters
    ----------
    node_coords : (n_nodes, ndim) 노드 좌표
    cell_node_list : 각 셀의 노드 인덱스 배열 리스트
    boundary_faces_dict : {경계이름: [면 노드 배열, ...]}
    cell_zone_map : {영역이름: [셀 인덱스, ...]}
    ndim : 공간 차원 (2 또는 3)
    cell_face_node_list : 3D용 - 각 셀의 면별 노드 리스트
                          [[face0_nodes, face1_nodes, ...], ...]

    Returns
    -------
    FVMesh
    """
    mesh = FVMesh(ndim=ndim)
    mesh.nodes = np.array(node_coords, dtype=np.float64)

    if ndim == 3 and cell_face_node_list is not None:
        return _build_fvmesh_3d(mesh, cell_node_list, cell_face_node_list,
                                boundary_faces_dict, cell_zone_map)

    # --- 2D 경로 (기존 로직) ---
    # 1) 셀 생성 및 기하 계산
    for i, cnodes in enumerate(cell_node_list):
        cnodes = np.array(cnodes, dtype=int)
        center, volume = _compute_cell_geometry_2d(mesh.nodes, cnodes)
        cell = Cell(nodes=cnodes, volume=volume, center=center)
        mesh.cells.append(cell)
    mesh.n_cells = len(mesh.cells)

    # 2) 에지(면) 추출 및 소유셀/이웃셀 결정
    edge_to_face_id: Dict[Tuple[int, int], int] = {}

    def _edge_key(n0, n1):
        return (min(n0, n1), max(n0, n1))

    # 내부면 구축
    for ci, cnodes_arr in enumerate(cell_node_list):
        cnodes_arr = np.array(cnodes_arr, dtype=int)
        n = len(cnodes_arr)
        for k in range(n):
            n0 = cnodes_arr[k]
            n1 = cnodes_arr[(k + 1) % n]
            ek = _edge_key(n0, n1)
            if ek in edge_to_face_id:
                fid = edge_to_face_id[ek]
                mesh.faces[fid].neighbour = ci
                mesh.cells[ci].faces.append(fid)
            else:
                face_nodes = np.array([n0, n1], dtype=int)
                center, normal, length = _compute_face_geometry_2d(mesh.nodes, face_nodes)
                # 법선이 소유셀에서 바깥쪽을 향하도록 보정
                to_face = center - mesh.cells[ci].center
                if np.dot(normal, to_face) < 0:
                    normal = -normal
                face = Face(
                    nodes=face_nodes, owner=ci, neighbour=-1,
                    area=length, normal=normal, center=center
                )
                fid = len(mesh.faces)
                mesh.faces.append(face)
                edge_to_face_id[ek] = fid
                mesh.cells[ci].faces.append(fid)

    # 3) 경계면 태깅
    boundary_edge_set: Dict[Tuple[int, int], str] = {}
    for bname, bface_list in boundary_faces_dict.items():
        for bf_nodes in bface_list:
            bf_nodes = np.array(bf_nodes, dtype=int)
            ek = _edge_key(bf_nodes[0], bf_nodes[1])
            boundary_edge_set[ek] = bname

    for fid, face in enumerate(mesh.faces):
        if face.neighbour == -1:
            ek = _edge_key(face.nodes[0], face.nodes[1])
            bname = boundary_edge_set.get(ek, "default")
            face.boundary_tag = bname
            if bname not in mesh.boundary_patches:
                mesh.boundary_patches[bname] = []
            mesh.boundary_patches[bname].append(fid)

    # 내부/경계면 수 집계
    mesh.n_faces = len(mesh.faces)
    mesh.n_internal_faces = sum(1 for f in mesh.faces if f.neighbour != -1)
    mesh.n_boundary_faces = mesh.n_faces - mesh.n_internal_faces

    # 셀 영역
    if cell_zone_map:
        mesh.cell_zones = cell_zone_map

    return mesh


def _build_fvmesh_3d(
    mesh: FVMesh,
    cell_node_list: List[np.ndarray],
    cell_face_node_list: List[List[np.ndarray]],
    boundary_faces_dict: Dict[str, List[np.ndarray]],
    cell_zone_map: Optional[Dict[str, List[int]]] = None
) -> FVMesh:
    """3D 메쉬 구성 (셀별 면 노드 리스트 사용)."""

    def _face_key(fnodes):
        return tuple(sorted(int(n) for n in fnodes))

    # 1) 면 추출 및 소유셀/이웃셀 결정
    face_key_to_id: Dict[tuple, int] = {}

    # 먼저 셀 객체를 노드 중심으로 생성 (체적은 나중에 계산)
    for ci, cnodes in enumerate(cell_node_list):
        cnodes = np.array(cnodes, dtype=int)
        center = np.mean(mesh.nodes[cnodes], axis=0)
        cell = Cell(nodes=cnodes, volume=0.0, center=center)
        mesh.cells.append(cell)
    mesh.n_cells = len(mesh.cells)

    # 면 생성
    for ci, face_list in enumerate(cell_face_node_list):
        for fnodes in face_list:
            fnodes = np.array(fnodes, dtype=int)
            fk = _face_key(fnodes)
            if fk in face_key_to_id:
                fid = face_key_to_id[fk]
                mesh.faces[fid].neighbour = ci
                mesh.cells[ci].faces.append(fid)
            else:
                center, normal, area = _compute_face_geometry_3d(mesh.nodes, fnodes)
                # 법선이 소유셀 바깥을 향하도록 보정
                to_face = center - mesh.cells[ci].center
                if np.dot(normal, to_face) < 0:
                    normal = -normal
                face = Face(
                    nodes=fnodes, owner=ci, neighbour=-1,
                    area=area, normal=normal, center=center
                )
                fid = len(mesh.faces)
                mesh.faces.append(face)
                face_key_to_id[fk] = fid
                mesh.cells[ci].faces.append(fid)

    # 2) 셀 체적 계산 (면이 모두 생성된 후)
    for ci, cell in enumerate(mesh.cells):
        _, volume = _compute_cell_geometry_3d(
            mesh.nodes, cell.nodes, cell.faces, mesh.faces)
        cell.volume = volume

    # 3) 경계면 태깅
    boundary_face_set: Dict[tuple, str] = {}
    for bname, bface_list in boundary_faces_dict.items():
        for bf_nodes in bface_list:
            bf_nodes = np.array(bf_nodes, dtype=int)
            fk = _face_key(bf_nodes)
            boundary_face_set[fk] = bname

    for fid, face in enumerate(mesh.faces):
        if face.neighbour == -1:
            fk = _face_key(face.nodes)
            bname = boundary_face_set.get(fk, "default")
            face.boundary_tag = bname
            if bname not in mesh.boundary_patches:
                mesh.boundary_patches[bname] = []
            mesh.boundary_patches[bname].append(fid)

    # 집계
    mesh.n_faces = len(mesh.faces)
    mesh.n_internal_faces = sum(1 for f in mesh.faces if f.neighbour != -1)
    mesh.n_boundary_faces = mesh.n_faces - mesh.n_internal_faces

    if cell_zone_map:
        mesh.cell_zones = cell_zone_map

    return mesh


def read_gmsh_msh(filepath: str, physical_names_map: Optional[Dict[int, str]] = None) -> FVMesh:
    """
    Gmsh .msh 파일을 읽어 FVMesh 객체로 변환.

    meshio 라이브러리를 사용하여 파일을 읽고, 2D 셀과 경계면을 추출한다.

    Parameters
    ----------
    filepath : .msh 파일 경로
    physical_names_map : {physical_tag: name} 매핑 (None이면 자동 추출)
    """
    import meshio

    msh = meshio.read(filepath)
    nodes_3d = msh.points

    # 차원 자동 감지: 3D 셀이 있으면 3D, 아니면 2D
    has_3d_cells = any(cb.type in ("tetra", "hexahedron", "wedge", "pyramid")
                       for cb in msh.cells)

    if has_3d_cells:
        ndim = 3
        node_coords = nodes_3d[:, :3]
        volume_types = ("tetra", "hexahedron", "wedge", "pyramid")
        surface_types = ("triangle", "quad")
    else:
        ndim = 2
        node_coords = nodes_3d[:, :2]
        volume_types = ("triangle", "quad")
        surface_types = ("line",)

    cell_node_list = []
    cell_tags = []

    boundary_lines = []
    boundary_line_tags = []

    for block_idx, cell_block in enumerate(msh.cells):
        ctype = cell_block.type
        if ctype in volume_types:
            for local_idx, cn in enumerate(cell_block.data):
                cell_node_list.append(cn)
                cell_tags.append((block_idx, local_idx))
        elif ctype in surface_types:
            for local_idx, ln in enumerate(cell_block.data):
                boundary_lines.append(ln)
                boundary_line_tags.append((block_idx, local_idx))

    # Physical names 처리
    if physical_names_map is None:
        physical_names_map = {}
        if hasattr(msh, 'field_data'):
            for name, (tag, dim) in msh.field_data.items():
                physical_names_map[tag] = name

    # 셀 데이터에서 physical tag 추출
    cell_zone_map: Dict[str, List[int]] = {}
    if msh.cell_data:
        for data_key in msh.cell_data:
            if 'physical' in data_key.lower() or 'gmsh:physical' in data_key.lower():
                break
        # gmsh:physical 태그
        phys_key = None
        for key in msh.cell_data:
            if 'physical' in key.lower():
                phys_key = key
                break

        if phys_key:
            cell_idx = 0
            for block_idx, cell_block in enumerate(msh.cells):
                tags = msh.cell_data[phys_key][block_idx]
                for local_idx, tag_val in enumerate(tags):
                    tag_val = int(tag_val)
                    if cell_block.type in ("triangle", "quad"):
                        zone_name = physical_names_map.get(tag_val, f"zone_{tag_val}")
                        if zone_name not in cell_zone_map:
                            cell_zone_map[zone_name] = []
                        cell_zone_map[zone_name].append(cell_idx)
                        cell_idx += 1

    # 경계면 분류
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}

    if msh.cell_data:
        phys_key = None
        for key in msh.cell_data:
            if 'physical' in key.lower():
                phys_key = key
                break

        if phys_key:
            line_block_count = 0
            for block_idx, cell_block in enumerate(msh.cells):
                if cell_block.type == "line":
                    tags = msh.cell_data[phys_key][block_idx]
                    for local_idx, tag_val in enumerate(tags):
                        tag_val = int(tag_val)
                        bname = physical_names_map.get(tag_val, f"boundary_{tag_val}")
                        if bname not in boundary_faces_dict:
                            boundary_faces_dict[bname] = []
                        boundary_faces_dict[bname].append(cell_block.data[local_idx])
                    line_block_count += 1

    # physical tag 없으면 모든 경계 line을 default로
    if not boundary_faces_dict and boundary_lines:
        boundary_faces_dict["default"] = [np.array(ln) for ln in boundary_lines]

    return build_fvmesh_from_arrays(node_coords, cell_node_list, boundary_faces_dict,
                                     cell_zone_map, ndim=ndim)
