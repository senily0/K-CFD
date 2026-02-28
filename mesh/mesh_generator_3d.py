"""
3D 구조 격자 생성기.

육면체(hex) 격자를 직접 생성한다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from mesh.mesh_reader import FVMesh, build_fvmesh_from_arrays


def generate_3d_channel_mesh(
    Lx: float = 1.0, Ly: float = 0.1, Lz: float = 0.1,
    nx: int = 20, ny: int = 10, nz: int = 10,
    boundary_names: Optional[Dict[str, str]] = None
) -> FVMesh:
    """
    3D 직육면체 채널 격자 생성 (구조 hex).

    Parameters
    ----------
    Lx, Ly, Lz : 각 방향 길이
    nx, ny, nz : 각 방향 셀 수
    boundary_names : 경계 이름 매핑
        기본값: inlet(x=0), outlet(x=Lx), wall_bottom(y=0),
                wall_top(y=Ly), wall_front(z=0), wall_back(z=Lz)

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

    # 노드 생성: (nx+1)*(ny+1)*(nz+1) 노드
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.zeros((n_nodes, 3))

    def nid(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = [i * dx, j * dy, k * dz]

    # 셀 생성: hex (8 노드)
    # hex 노드 순서: 하면 반시계(n0,n1,n2,n3), 상면 반시계(n4,n5,n6,n7)
    cell_node_list = []
    cell_face_node_list = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # 8 꼭짓점
                n0 = nid(i, j, k)
                n1 = nid(i + 1, j, k)
                n2 = nid(i + 1, j + 1, k)
                n3 = nid(i, j + 1, k)
                n4 = nid(i, j, k + 1)
                n5 = nid(i + 1, j, k + 1)
                n6 = nid(i + 1, j + 1, k + 1)
                n7 = nid(i, j + 1, k + 1)

                cell_nodes = np.array([n0, n1, n2, n3, n4, n5, n6, n7])
                cell_node_list.append(cell_nodes)

                # 6면 (각 면의 노드, 외향 법선 기준으로 반시계)
                faces = [
                    np.array([n0, n3, n2, n1]),  # z- (bottom, k면)
                    np.array([n4, n5, n6, n7]),  # z+ (top, k+1면)
                    np.array([n0, n1, n5, n4]),  # y- (front, j면)
                    np.array([n2, n3, n7, n6]),  # y+ (back, j+1면)
                    np.array([n0, n4, n7, n3]),  # x- (left, i면)
                    np.array([n1, n2, n6, n5]),  # x+ (right, i+1면)
                ]
                cell_face_node_list.append(faces)

    # 경계면 생성
    boundary_faces_dict: Dict[str, List[np.ndarray]] = {}

    # x=0 (inlet)
    bname = boundary_names.get('x_min', 'inlet')
    boundary_faces_dict[bname] = []
    for k in range(nz):
        for j in range(ny):
            n0 = nid(0, j, k)
            n3 = nid(0, j + 1, k)
            n7 = nid(0, j + 1, k + 1)
            n4 = nid(0, j, k + 1)
            boundary_faces_dict[bname].append(np.array([n0, n4, n7, n3]))

    # x=Lx (outlet)
    bname = boundary_names.get('x_max', 'outlet')
    boundary_faces_dict[bname] = []
    for k in range(nz):
        for j in range(ny):
            n1 = nid(nx, j, k)
            n2 = nid(nx, j + 1, k)
            n6 = nid(nx, j + 1, k + 1)
            n5 = nid(nx, j, k + 1)
            boundary_faces_dict[bname].append(np.array([n1, n2, n6, n5]))

    # y=0 (wall_bottom)
    bname = boundary_names.get('y_min', 'wall_bottom')
    boundary_faces_dict[bname] = []
    for k in range(nz):
        for i in range(nx):
            n0 = nid(i, 0, k)
            n1 = nid(i + 1, 0, k)
            n5 = nid(i + 1, 0, k + 1)
            n4 = nid(i, 0, k + 1)
            boundary_faces_dict[bname].append(np.array([n0, n1, n5, n4]))

    # y=Ly (wall_top)
    bname = boundary_names.get('y_max', 'wall_top')
    boundary_faces_dict[bname] = []
    for k in range(nz):
        for i in range(nx):
            n3 = nid(i, ny, k)
            n2 = nid(i + 1, ny, k)
            n6 = nid(i + 1, ny, k + 1)
            n7 = nid(i, ny, k + 1)
            boundary_faces_dict[bname].append(np.array([n2, n3, n7, n6]))

    # z=0 (wall_front)
    bname = boundary_names.get('z_min', 'wall_front')
    boundary_faces_dict[bname] = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j, 0)
            n1 = nid(i + 1, j, 0)
            n2 = nid(i + 1, j + 1, 0)
            n3 = nid(i, j + 1, 0)
            boundary_faces_dict[bname].append(np.array([n0, n3, n2, n1]))

    # z=Lz (wall_back)
    bname = boundary_names.get('z_max', 'wall_back')
    boundary_faces_dict[bname] = []
    for j in range(ny):
        for i in range(nx):
            n4 = nid(i, j, nz)
            n5 = nid(i + 1, j, nz)
            n6 = nid(i + 1, j + 1, nz)
            n7 = nid(i, j + 1, nz)
            boundary_faces_dict[bname].append(np.array([n4, n5, n6, n7]))

    return build_fvmesh_from_arrays(
        nodes, cell_node_list, boundary_faces_dict,
        ndim=3, cell_face_node_list=cell_face_node_list
    )


def generate_3d_duct_mesh(
    Lx: float = 2.0, Ly: float = 0.1, Lz: float = 0.1,
    nx: int = 20, ny: int = 10, nz: int = 10
) -> FVMesh:
    """
    3D 정사각 덕트 격자 (Poiseuille 검증용).

    경계: inlet(x=0), outlet(x=Lx), 나머지 4면 = wall
    """
    return generate_3d_channel_mesh(
        Lx, Ly, Lz, nx, ny, nz,
        boundary_names={
            'x_min': 'inlet', 'x_max': 'outlet',
            'y_min': 'wall_bottom', 'y_max': 'wall_top',
            'z_min': 'wall_front', 'z_max': 'wall_back'
        }
    )


def generate_3d_cavity_mesh(
    Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
    nx: int = 16, ny: int = 16, nz: int = 16
) -> FVMesh:
    """
    3D 뚜껑 구동 공동(Lid-Driven Cavity) 격자 생성.

    경계:
      - lid      (y=Ly): 이동 벽 (상부 덮개)
      - wall_bottom (y=0)
      - wall_left   (x=0)
      - wall_right  (x=Lx)
      - wall_front  (z=0)
      - wall_back   (z=Lz)
    """
    return generate_3d_channel_mesh(
        Lx, Ly, Lz, nx, ny, nz,
        boundary_names={
            'x_min': 'wall_left',
            'x_max': 'wall_right',
            'y_min': 'wall_bottom',
            'y_max': 'lid',
            'z_min': 'wall_front',
            'z_max': 'wall_back',
        }
    )
