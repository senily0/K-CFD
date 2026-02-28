"""
Gmsh Python API를 사용한 검증용 격자 생성기.

각 검증 케이스에 필요한 2D 격자를 생성한다.
"""

import numpy as np
import gmsh
from typing import Tuple, Optional, Dict, List
from mesh.mesh_reader import FVMesh, build_fvmesh_from_arrays


def _make_structured_quad_mesh(
    x0: float, y0: float, x1: float, y1: float,
    nx: int, ny: int,
    boundary_names: Dict[str, str] = None
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, List[np.ndarray]]]:
    """
    직사각형 영역에 구조 격자(quad) 생성. Gmsh 없이 직접 생성.

    Parameters
    ----------
    x0, y0 : 왼쪽 하단 좌표
    x1, y1 : 오른쪽 상단 좌표
    nx, ny : x, y 방향 셀 수
    boundary_names : {'bottom','top','left','right'} 이름 매핑

    Returns
    -------
    nodes, cell_node_list, boundary_faces_dict
    """
    if boundary_names is None:
        boundary_names = {
            'bottom': 'bottom', 'top': 'top',
            'left': 'left', 'right': 'right'
        }

    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny

    # 노드 생성
    nodes = np.zeros(((nx + 1) * (ny + 1), 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = j * (nx + 1) + i
            nodes[nid] = [x0 + i * dx, y0 + j * dy]

    # 셀 생성 (quad: 4 노드, 반시계 방향)
    cell_node_list = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            cell_node_list.append(np.array([n0, n1, n2, n3]))

    # 경계면
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}

    # Bottom (j=0)
    bname = boundary_names.get('bottom', 'bottom')
    boundary_faces_dict[bname] = []
    for i in range(nx):
        n0 = i
        n1 = i + 1
        boundary_faces_dict[bname].append(np.array([n0, n1]))

    # Top (j=ny)
    bname = boundary_names.get('top', 'top')
    boundary_faces_dict[bname] = []
    for i in range(nx):
        n0 = ny * (nx + 1) + i
        n1 = n0 + 1
        boundary_faces_dict[bname].append(np.array([n0, n1]))

    # Left (i=0)
    bname = boundary_names.get('left', 'left')
    boundary_faces_dict[bname] = []
    for j in range(ny):
        n0 = j * (nx + 1)
        n1 = (j + 1) * (nx + 1)
        boundary_faces_dict[bname].append(np.array([n0, n1]))

    # Right (i=nx)
    bname = boundary_names.get('right', 'right')
    boundary_faces_dict[bname] = []
    for j in range(ny):
        n0 = j * (nx + 1) + nx
        n1 = (j + 1) * (nx + 1) + nx
        boundary_faces_dict[bname].append(np.array([n0, n1]))

    return nodes, cell_node_list, boundary_faces_dict


def generate_channel_mesh(length: float = 1.0, height: float = 0.1,
                          nx: int = 50, ny: int = 20) -> FVMesh:
    """
    Poiseuille 유동 검증을 위한 2D 채널 격자.

    경계: inlet(left), outlet(right), wall_bottom, wall_top
    """
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, length, height, nx, ny,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'wall_top',
            'left': 'inlet', 'right': 'outlet'
        }
    )
    return build_fvmesh_from_arrays(nodes, cells, bfaces)


def generate_cavity_mesh(size: float = 1.0, n: int = 64) -> FVMesh:
    """
    Lid-driven cavity 검증을 위한 2D 정사각형 격자.

    경계: lid(top), wall_bottom, wall_left, wall_right
    """
    nodes, cells, bfaces = _make_structured_quad_mesh(
        0.0, 0.0, size, size, n, n,
        boundary_names={
            'bottom': 'wall_bottom', 'top': 'lid',
            'left': 'wall_left', 'right': 'wall_right'
        }
    )
    return build_fvmesh_from_arrays(nodes, cells, bfaces)


def generate_cht_mesh(
    fluid_length: float = 1.0, fluid_height: float = 0.05,
    solid_height: float = 0.01,
    nx: int = 80, ny_fluid: int = 20, ny_solid: int = 8
) -> Tuple[FVMesh, Dict[str, List[int]]]:
    """
    CHT 검증을 위한 유체+고체 멀티존 격자.

    고체 영역이 유체 아래에 위치. 인터페이스는 y=0 평면.

    Returns
    -------
    mesh : 전체 FVMesh
    zone_map : {'fluid': [셀인덱스], 'solid': [셀인덱스]}
    """
    # 고체 영역: y = [-solid_height, 0]
    # 유체 영역: y = [0, fluid_height]
    total_height = solid_height + fluid_height
    ny_total = ny_solid + ny_fluid

    dx = fluid_length / nx
    dy_solid = solid_height / ny_solid
    dy_fluid = fluid_height / ny_fluid

    # 노드 생성
    n_nodes = (nx + 1) * (ny_total + 1)
    nodes = np.zeros((n_nodes, 2))
    for j in range(ny_total + 1):
        if j <= ny_solid:
            y = -solid_height + j * dy_solid
        else:
            y = (j - ny_solid) * dy_fluid
        for i in range(nx + 1):
            nid = j * (nx + 1) + i
            nodes[nid] = [i * dx, y]

    # 셀 생성
    cell_node_list = []
    solid_cells = []
    fluid_cells = []
    cell_idx = 0
    for j in range(ny_total):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            cell_node_list.append(np.array([n0, n1, n2, n3]))
            if j < ny_solid:
                solid_cells.append(cell_idx)
            else:
                fluid_cells.append(cell_idx)
            cell_idx += 1

    zone_map = {'fluid': fluid_cells, 'solid': solid_cells}

    # 경계면
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}

    # Bottom (고체 하면)
    boundary_faces_dict['wall_heated'] = []
    for i in range(nx):
        n0 = i
        n1 = i + 1
        boundary_faces_dict['wall_heated'].append(np.array([n0, n1]))

    # Top (유체 상면)
    boundary_faces_dict['wall_top'] = []
    for i in range(nx):
        n0 = ny_total * (nx + 1) + i
        n1 = n0 + 1
        boundary_faces_dict['wall_top'].append(np.array([n0, n1]))

    # Left
    boundary_faces_dict['inlet'] = []
    for j in range(ny_solid, ny_total):  # 유체부만 inlet
        n0 = j * (nx + 1)
        n1 = (j + 1) * (nx + 1)
        boundary_faces_dict['inlet'].append(np.array([n0, n1]))

    boundary_faces_dict['wall_left_solid'] = []
    for j in range(ny_solid):
        n0 = j * (nx + 1)
        n1 = (j + 1) * (nx + 1)
        boundary_faces_dict['wall_left_solid'].append(np.array([n0, n1]))

    # Right
    boundary_faces_dict['outlet'] = []
    for j in range(ny_solid, ny_total):
        n0 = j * (nx + 1) + nx
        n1 = (j + 1) * (nx + 1) + nx
        boundary_faces_dict['outlet'].append(np.array([n0, n1]))

    boundary_faces_dict['wall_right_solid'] = []
    for j in range(ny_solid):
        n0 = j * (nx + 1) + nx
        n1 = (j + 1) * (nx + 1) + nx
        boundary_faces_dict['wall_right_solid'].append(np.array([n0, n1]))

    mesh = build_fvmesh_from_arrays(nodes, cell_node_list, boundary_faces_dict, zone_map)
    return mesh, zone_map


def generate_bubble_column_mesh(width: float = 0.15, height: float = 0.45,
                                 nx: int = 30, ny: int = 90) -> FVMesh:
    """
    기포탑 검증을 위한 2D 직사각형 격자.

    경계: inlet_gas(bottom 중앙 1/3), wall_bottom(나머지 bottom),
          outlet(top), wall_left, wall_right
    """
    nodes, cells, bfaces_raw = _make_structured_quad_mesh(
        0.0, 0.0, width, height, nx, ny,
        boundary_names={
            'bottom': '_bottom_temp', 'top': 'outlet',
            'left': 'wall_left', 'right': 'wall_right'
        }
    )

    # bottom을 inlet_gas(중앙 1/3)와 wall_bottom으로 분할
    bottom_faces = bfaces_raw.pop('_bottom_temp')
    bfaces_raw['inlet_gas'] = []
    bfaces_raw['wall_bottom'] = []

    x_left = width / 3.0
    x_right = 2.0 * width / 3.0

    for bf in bottom_faces:
        xc = 0.5 * (nodes[bf[0], 0] + nodes[bf[1], 0])
        if x_left <= xc <= x_right:
            bfaces_raw['inlet_gas'].append(bf)
        else:
            bfaces_raw['wall_bottom'].append(bf)

    return build_fvmesh_from_arrays(nodes, cells, bfaces_raw)


def generate_triangle_channel_mesh(
    Lx: float = 1.0, Ly: float = 0.1,
    nx: int = 20, ny: int = 5
) -> FVMesh:
    """
    Poiseuille 유동 검증을 위한 2D 채널 삼각형 격자.

    구조적 쿼드 격자를 대각선(n0-n2)으로 분할하여 삼각형 셀 생성.

    경계: inlet(left), outlet(right), wall_bottom, wall_top
    """
    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    dx = Lx / nx
    dy = Ly / ny

    # 노드 생성
    nodes = np.zeros(((nx + 1) * (ny + 1), 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[nid(i, j)] = [i * dx, j * dy]

    # 셀 생성: 각 쿼드를 대각선(n0-n2)으로 2개의 삼각형으로 분할
    cell_node_list = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i,     j)      # bottom-left
            n1 = nid(i + 1, j)      # bottom-right
            n2 = nid(i + 1, j + 1)  # top-right
            n3 = nid(i,     j + 1)  # top-left
            # Triangle A (lower-right): n0, n1, n2
            cell_node_list.append(np.array([n0, n1, n2]))
            # Triangle B (upper-left): n0, n2, n3
            cell_node_list.append(np.array([n0, n2, n3]))

    # 경계면
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}

    # inlet (x=0, left)
    boundary_faces_dict['inlet'] = []
    for j in range(ny):
        boundary_faces_dict['inlet'].append(np.array([nid(0, j), nid(0, j + 1)]))

    # outlet (x=Lx, right)
    boundary_faces_dict['outlet'] = []
    for j in range(ny):
        boundary_faces_dict['outlet'].append(np.array([nid(nx, j), nid(nx, j + 1)]))

    # wall_bottom (y=0)
    boundary_faces_dict['wall_bottom'] = []
    for i in range(nx):
        boundary_faces_dict['wall_bottom'].append(np.array([nid(i, 0), nid(i + 1, 0)]))

    # wall_top (y=Ly)
    boundary_faces_dict['wall_top'] = []
    for i in range(nx):
        boundary_faces_dict['wall_top'].append(np.array([nid(i, ny), nid(i + 1, ny)]))

    return build_fvmesh_from_arrays(nodes, cell_node_list, boundary_faces_dict)
