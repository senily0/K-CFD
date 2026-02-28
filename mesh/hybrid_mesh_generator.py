"""
혼합 격자(Hex/Tet) 생성기.

육면체(hex) 영역과 사면체(tet) 영역을 조합한 3D 혼합 격자를 생성한다.
Tet 영역은 중심점 삽입 방식의 24-tet 분해를 사용하여
인접 셀 간 면 매칭을 보장한다.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from mesh.mesh_reader import FVMesh, build_fvmesh_from_arrays


def _split_quad_consistent(quad_nodes) -> List[Tuple[int, int, int]]:
    """
    사각형을 2개 삼각형으로 일관되게 분할.
    최소 노드 인덱스 기반 대각선 선택으로 인접 셀 매칭 보장.

    Returns list of (a,b,c) tuples.
    """
    a, b, c, d = [int(x) for x in quad_nodes]
    m = min(a, b, c, d)
    if m == a or m == c:
        return [(a, b, c), (a, c, d)]
    else:
        return [(a, b, d), (b, c, d)]


def _hex_to_tets_with_center(hex_nodes: np.ndarray, center_id: int) -> List[np.ndarray]:
    """
    육면체 → 사면체 분해 (중심점 삽입).

    각 사각형 면을 2개 삼각형으로 분할(최소 노드 대각선)한 후,
    각 삼각형과 중심점으로 사면체를 생성한다.

    이 방법은 인접 셀과 공유 면이 동일한 삼각형 분할을 가짐을 보장한다
    (중심점은 셀 내부이므로 공유 면에 영향을 주지 않음).

    hex_nodes: [n0,n1,n2,n3,n4,n5,n6,n7]
    center_id: 중심점 노드 인덱스
    Returns: 12개 사면체 (6면 × 2삼각형)
    """
    n = [int(x) for x in hex_nodes]

    # 6개 면 (외향 법선 기준 반시계 순서)
    faces = [
        (n[0], n[3], n[2], n[1]),  # bottom (z-)
        (n[4], n[5], n[6], n[7]),  # top (z+)
        (n[0], n[1], n[5], n[4]),  # front (y-)
        (n[2], n[3], n[7], n[6]),  # back (y+)
        (n[0], n[4], n[7], n[3]),  # left (x-)
        (n[1], n[2], n[6], n[5]),  # right (x+)
    ]

    tets = []
    for quad in faces:
        for tri in _split_quad_consistent(quad):
            a, b, c = tri
            tets.append(np.array([a, b, c, center_id], dtype=int))

    return tets


def _tet_faces(tet_nodes: np.ndarray) -> List[np.ndarray]:
    """사면체의 4개 삼각형 면 반환."""
    n0, n1, n2, n3 = tet_nodes
    return [
        np.array([n0, n2, n1], dtype=int),
        np.array([n0, n1, n3], dtype=int),
        np.array([n1, n2, n3], dtype=int),
        np.array([n0, n3, n2], dtype=int),
    ]


def _hex_faces(hex_nodes: np.ndarray) -> List[np.ndarray]:
    """육면체의 6개 사각형 면 반환."""
    n0, n1, n2, n3, n4, n5, n6, n7 = hex_nodes
    return [
        np.array([n0, n3, n2, n1], dtype=int),  # z- (bottom)
        np.array([n4, n5, n6, n7], dtype=int),  # z+ (top)
        np.array([n0, n1, n5, n4], dtype=int),  # y- (front)
        np.array([n2, n3, n7, n6], dtype=int),  # y+ (back)
        np.array([n0, n4, n7, n3], dtype=int),  # x- (left)
        np.array([n1, n2, n6, n5], dtype=int),  # x+ (right)
    ]


def _face_key(nodes):
    """면 노드 키 (정렬된 튜플)."""
    return tuple(sorted(int(n) for n in nodes))


def generate_hybrid_hex_tet_mesh(
    Lx: float = 2.0, Ly: float = 0.1, Lz: float = 0.1,
    nx: int = 20, ny: int = 8, nz: int = 8,
    tet_fraction: float = 0.5,
    boundary_names: Optional[Dict[str, str]] = None
) -> FVMesh:
    """
    Hex/Tet 혼합 격자 생성.

    x 방향으로 hex 영역(입구 측)과 tet 영역(출구 측)으로 분할한다.
    Tet 영역은 중심점 삽입 + 최소노드 대각선 분해를 사용한다.

    Parameters
    ----------
    Lx, Ly, Lz : 각 방향 길이 [m]
    nx, ny, nz : 각 방향 셀 수
    tet_fraction : tet 영역 비율 (0~1, x 방향)
    boundary_names : 경계 이름 매핑

    Returns
    -------
    FVMesh (ndim=3)
    """
    if boundary_names is None:
        boundary_names = {
            'x_min': 'inlet', 'x_max': 'outlet',
            'y_min': 'wall_bottom', 'y_max': 'wall_top',
            'z_min': 'wall_front', 'z_max': 'wall_back'
        }

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    nx_hex = max(1, int(nx * (1.0 - tet_fraction)))
    nx_tet = nx - nx_hex

    # 기본 구조 격자 노드: (nx+1)*(ny+1)*(nz+1)
    n_base_nodes = (nx + 1) * (ny + 1) * (nz + 1)

    def nid(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    # 추가 중심점: tet 영역의 각 hex 셀마다 1개
    n_tet_cells_hex = nx_tet * ny * nz
    n_total_nodes = n_base_nodes + n_tet_cells_hex

    nodes = np.zeros((n_total_nodes, 3))

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = [i * dx, j * dy, k * dz]

    # 중심점 좌표 계산
    center_node_id = n_base_nodes
    center_id_map = {}  # (i,j,k) -> center node id

    for k in range(nz):
        for j in range(ny):
            for i in range(nx_hex, nx):
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                cz = (k + 0.5) * dz
                nodes[center_node_id] = [cx, cy, cz]
                center_id_map[(i, j, k)] = center_node_id
                center_node_id += 1

    # 경계 노드 집합 사전 계산
    tol = 1e-10
    boundary_node_sets: Dict[str, Set[int]] = {
        'x_min': set(), 'x_max': set(),
        'y_min': set(), 'y_max': set(),
        'z_min': set(), 'z_max': set(),
    }
    for nid_val in range(n_base_nodes):
        x, y, z = nodes[nid_val]
        if abs(x) < tol:
            boundary_node_sets['x_min'].add(nid_val)
        if abs(x - Lx) < tol:
            boundary_node_sets['x_max'].add(nid_val)
        if abs(y) < tol:
            boundary_node_sets['y_min'].add(nid_val)
        if abs(y - Ly) < tol:
            boundary_node_sets['y_max'].add(nid_val)
        if abs(z) < tol:
            boundary_node_sets['z_min'].add(nid_val)
        if abs(z - Lz) < tol:
            boundary_node_sets['z_max'].add(nid_val)

    cell_node_list = []
    cell_face_node_list = []
    cell_zone_map = {'hex': [], 'tet': []}
    face_cell_count: Dict[Tuple, int] = {}
    cell_idx = 0

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = nid(i, j, k)
                n1 = nid(i + 1, j, k)
                n2 = nid(i + 1, j + 1, k)
                n3 = nid(i, j + 1, k)
                n4 = nid(i, j, k + 1)
                n5 = nid(i + 1, j, k + 1)
                n6 = nid(i + 1, j + 1, k + 1)
                n7 = nid(i, j + 1, k + 1)

                hex_nodes_arr = np.array([n0, n1, n2, n3, n4, n5, n6, n7], dtype=int)

                if i < nx_hex:
                    if i == nx_hex - 1 and nx_tet > 0:
                        # 인터페이스 hex: x+ 면을 삼각형으로 분할
                        faces = _hex_faces(hex_nodes_arr)
                        x_plus_quad = faces[5]  # [n1, n2, n6, n5]
                        tris_raw = _split_quad_consistent(x_plus_quad)
                        tris = [np.array(t, dtype=int) for t in tris_raw]
                        faces_new = faces[:5] + tris
                        cell_node_list.append(hex_nodes_arr)
                        cell_face_node_list.append(faces_new)
                        for fn in faces_new:
                            fk = _face_key(fn)
                            face_cell_count[fk] = face_cell_count.get(fk, 0) + 1
                    else:
                        faces = _hex_faces(hex_nodes_arr)
                        cell_node_list.append(hex_nodes_arr)
                        cell_face_node_list.append(faces)
                        for fn in faces:
                            fk = _face_key(fn)
                            face_cell_count[fk] = face_cell_count.get(fk, 0) + 1
                    cell_zone_map['hex'].append(cell_idx)
                    cell_idx += 1
                else:
                    # Tet 영역: 중심점 기반 12-tet 분해
                    cid = center_id_map[(i, j, k)]
                    tets = _hex_to_tets_with_center(hex_nodes_arr, cid)
                    for tet in tets:
                        faces = _tet_faces(tet)
                        cell_node_list.append(tet)
                        cell_face_node_list.append(faces)
                        for fn in faces:
                            fk = _face_key(fn)
                            face_cell_count[fk] = face_cell_count.get(fk, 0) + 1
                        cell_zone_map['tet'].append(cell_idx)
                        cell_idx += 1

    # 경계면 식별
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}
    for bkey in boundary_names:
        bname = boundary_names[bkey]
        boundary_faces_dict[bname] = []

    seen_boundary = set()
    for ci, face_list in enumerate(cell_face_node_list):
        for fn in face_list:
            fk = _face_key(fn)
            if fk in seen_boundary:
                continue
            if face_cell_count.get(fk, 0) == 1:
                # 중심점을 포함하는 면은 내부 면 (경계에 있을 수 없음)
                fn_ints = [int(n) for n in fn]
                if any(n >= n_base_nodes for n in fn_ints):
                    continue
                seen_boundary.add(fk)
                fn_set = set(fn_ints)
                for bkey, node_set in boundary_node_sets.items():
                    if fn_set.issubset(node_set):
                        bname = boundary_names.get(bkey, bkey)
                        if bname not in boundary_faces_dict:
                            boundary_faces_dict[bname] = []
                        boundary_faces_dict[bname].append(fn)
                        break

    return build_fvmesh_from_arrays(
        nodes, cell_node_list, boundary_faces_dict,
        cell_zone_map=cell_zone_map,
        ndim=3, cell_face_node_list=cell_face_node_list
    )
