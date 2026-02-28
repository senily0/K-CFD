"""
ParaView 호환 VTU 파일 내보내기.

meshio를 사용하여 FVMesh를 VTU(Unstructured Grid) 형식으로 변환.
2D 격자는 z=0 평면의 3D 좌표로 확장하여 ParaView에서 시각화 가능.
"""

import numpy as np

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


def export_mesh_to_vtu(mesh, filepath: str, cell_data: dict = None):
    """
    FVMesh를 ParaView 호환 VTU 파일로 내보내기.

    Parameters
    ----------
    mesh : FVMesh
        내보낼 격자
    filepath : str
        출력 VTU 파일 경로 (.vtu)
    cell_data : dict, optional
        셀 데이터 {'field_name': np.ndarray(n_cells,), ...}
    """
    if not HAS_MESHIO:
        print("  [VTU] meshio 미설치 - VTU 내보내기 건너뜀")
        return

    ndim = getattr(mesh, 'ndim', 2)
    nodes = mesh.nodes

    # 2D 격자: z=0 좌표 추가하여 3D로 확장
    if ndim == 2 or nodes.shape[1] == 2:
        points = np.zeros((nodes.shape[0], 3))
        points[:, :2] = nodes[:, :2]
    else:
        points = nodes[:, :3].copy()
        if nodes.shape[1] < 3:
            pts = np.zeros((nodes.shape[0], 3))
            pts[:, :nodes.shape[1]] = nodes
            points = pts

    # 셀 타입별 분류
    cell_blocks = {}
    cell_order = []  # 원래 셀 인덱스 순서 유지

    for ci in range(mesh.n_cells):
        cell = mesh.cells[ci]
        n_nodes = len(cell.nodes)

        if ndim == 2 or nodes.shape[1] == 2:
            if n_nodes == 4:
                ctype = 'quad'
            elif n_nodes == 3:
                ctype = 'triangle'
            else:
                ctype = 'polygon'
        else:
            if n_nodes == 8:
                ctype = 'hexahedron'
            elif n_nodes == 4:
                ctype = 'tetra'
            elif n_nodes == 6:
                ctype = 'wedge'
            elif n_nodes == 5:
                ctype = 'pyramid'
            else:
                ctype = 'polyhedron'

        if ctype not in cell_blocks:
            cell_blocks[ctype] = []
        cell_blocks[ctype].append((ci, cell.nodes))

    # meshio 셀 블록 구성
    meshio_cells = []
    index_map = np.zeros(mesh.n_cells, dtype=int)  # 원래 → meshio 순서

    global_idx = 0
    for ctype, cells_list in cell_blocks.items():
        connectivity = np.array([c[1] for c in cells_list], dtype=np.intp)
        meshio_cells.append((ctype, connectivity))
        for orig_idx, _ in cells_list:
            index_map[orig_idx] = global_idx
            global_idx += 1

    # 셀 데이터 재정렬 (meshio 순서로)
    meshio_cell_data = {}
    if cell_data:
        for name, values in cell_data.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape[0] != mesh.n_cells:
                continue
            # meshio 순서로 재정렬
            reordered = np.zeros_like(arr)
            for orig_i in range(mesh.n_cells):
                reordered[index_map[orig_i]] = arr[orig_i]

            # 셀 블록별 분할
            block_data = []
            offset = 0
            for ctype, cells_list in cell_blocks.items():
                n = len(cells_list)
                block_data.append(reordered[offset:offset + n])
                offset += n
            meshio_cell_data[name] = block_data

    # meshio Mesh 생성 및 저장
    m = meshio.Mesh(points, meshio_cells, cell_data=meshio_cell_data)
    m.write(filepath)


def export_input_json(params: dict, filepath: str):
    """
    입력 파라미터를 JSON으로 저장.

    Parameters
    ----------
    params : dict
        입력 파라미터 딕셔너리
    filepath : str
        출력 JSON 파일 경로
    """
    import json

    # numpy 타입을 Python 기본 타입으로 변환
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(params), f, indent=2, ensure_ascii=False)
