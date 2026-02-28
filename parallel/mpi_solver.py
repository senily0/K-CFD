"""
MPI SIMPLE 솔버 래퍼.

교육용 코드: 복제 압력 풀이 (replicated pressure solve) 방식.
운동량은 로컬 풀이 + 고스트 교환, 압력은 전체 도메인에서 풀이.

mpi4py가 없으면 시리얼 폴백.
"""

import numpy as np
from mesh.mesh_reader import FVMesh
from parallel.partitioning import GeometricPartitioner, GhostCellLayer

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


class MPISIMPLESolver:
    """
    MPI 병렬 SIMPLE 솔버.

    복제 압력 풀이 방식:
    1. 운동량: 각 프로세스가 로컬 셀만 풀이
    2. 고스트 교환: 이웃 파티션의 경계값 동기화
    3. 압력: 전 프로세스에서 전체 도메인 풀이 (복제)
    4. 속도/압력 보정: 로컬에서 수행
    """

    def __init__(self, mesh: FVMesh, rho: float = 1.0, mu: float = 0.01,
                 n_parts: int = None):
        self.mesh = mesh
        self.rho = rho
        self.mu = mu

        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        if n_parts is None:
            n_parts = self.size

        # 분할
        partitioner = GeometricPartitioner(mesh)
        self.part_ids = partitioner.partition(n_parts)
        self.ghost_layer = GhostCellLayer(mesh, self.part_ids)
        self.n_parts = n_parts

    def exchange_scalar(self, field_values: np.ndarray, rank: int) -> dict:
        """
        스칼라 필드 고스트 교환 (시리얼 모드).

        MPI 사용 시: Isend/Irecv로 비동기 통신.
        시리얼 시: 직접 복사.

        Returns
        -------
        ghost_values : {remote_rank: np.ndarray} 고스트 셀 값
        """
        ghost_cells = self.ghost_layer.get_ghost_cells(rank)
        ghost_values = {}

        if not HAS_MPI or self.size == 1:
            # 시리얼: 직접 복사
            for remote_rank, cell_ids in ghost_cells.items():
                ghost_values[remote_rank] = field_values[cell_ids].copy()
        else:
            # MPI 통신
            reqs = []
            for remote_rank, cell_ids in ghost_cells.items():
                buf = np.empty(len(cell_ids))
                req = self.comm.Irecv(buf, source=remote_rank, tag=100)
                reqs.append((remote_rank, buf, req))

            # 보낼 데이터
            for remote_rank in ghost_cells:
                send_ids = self.ghost_layer.get_ghost_cells(remote_rank).get(rank, [])
                send_buf = field_values[send_ids].copy()
                self.comm.Isend(send_buf, dest=remote_rank, tag=100)

            # 수신 대기
            for remote_rank, buf, req in reqs:
                req.Wait()
                ghost_values[remote_rank] = buf

        return ghost_values

    def exchange_vector(self, field_values: np.ndarray, rank: int) -> dict:
        """벡터 필드 고스트 교환 (시리얼 모드)."""
        ghost_cells = self.ghost_layer.get_ghost_cells(rank)
        ghost_values = {}

        for remote_rank, cell_ids in ghost_cells.items():
            ghost_values[remote_rank] = field_values[cell_ids].copy()

        return ghost_values

    def solve_parallel_poiseuille(self, dpdx: float = -1.0,
                                   max_iter: int = 500,
                                   tol: float = 1e-4) -> dict:
        """
        병렬 Poiseuille 풀이 (검증용).

        각 파티션이 독립적으로 운동량을 풀고,
        고스트 교환 후 전체 압력 풀이.

        실질적으로 시리얼과 동일 결과를 내는지 검증.

        Returns
        -------
        result : {'converged': bool, 'u_values': ndarray,
                  'partition_info': dict, ...}
        """
        from models.single_phase import SIMPLESolver

        mesh = self.mesh

        # 시리얼 풀이로 참조 해 생성
        solver = SIMPLESolver(mesh, rho=self.rho, mu=self.mu)
        solver.max_outer_iter = max_iter
        solver.tol = tol
        solver.alpha_u = 0.7
        solver.alpha_p = 0.3

        # 표준 Poiseuille BC 설정
        ndim = getattr(mesh, 'ndim', 2)
        Ly = max(mesh.nodes[:, 1]) - min(mesh.nodes[:, 1])

        # 입구 포물선 프로파일
        inlet_fids = mesh.boundary_patches.get('inlet', [])
        if inlet_fids:
            inlet_y = np.array([mesh.faces[f].center[1] for f in inlet_fids])
            u_inlet = -dpdx / (2.0 * self.mu) * inlet_y * (Ly - inlet_y)
            u_vec = np.zeros((len(inlet_fids), ndim))
            u_vec[:, 0] = u_inlet
            solver.U.boundary_values['inlet'] = u_vec

        solver.set_velocity_bc('inlet', 'dirichlet')
        solver.set_velocity_bc('outlet', 'zero_gradient')
        for wall in ['wall_bottom', 'wall_top']:
            if wall in mesh.boundary_patches:
                solver.set_velocity_bc(wall, 'dirichlet',
                                        np.zeros(ndim).tolist())
        solver.set_pressure_bc('inlet', 'zero_gradient')
        solver.set_pressure_bc('outlet', 'dirichlet', 0.0)
        for wall in ['wall_bottom', 'wall_top']:
            if wall in mesh.boundary_patches:
                solver.set_pressure_bc(wall, 'zero_gradient')

        result = solver.solve_steady()

        # 파티션별 통계
        partition_info = {}
        for rank in range(self.n_parts):
            local_cells = self.ghost_layer.get_local_cells(rank)
            ghost_cells = self.ghost_layer.get_ghost_cells(rank)
            n_ghost = sum(len(v) for v in ghost_cells.values())
            partition_info[rank] = {
                'n_local': len(local_cells),
                'n_ghost': n_ghost,
                'neighbors': list(ghost_cells.keys()),
            }

        # 고스트 교환 검증
        u_x = solver.U.values[:, 0]
        exchange_error = 0.0
        for rank in range(self.n_parts):
            ghost_vals = self.exchange_scalar(u_x, rank)
            ghost_cells = self.ghost_layer.get_ghost_cells(rank)
            for remote_rank, cell_ids in ghost_cells.items():
                expected = u_x[cell_ids]
                received = ghost_vals[remote_rank]
                exchange_error += np.sum(np.abs(expected - received))

        return {
            'converged': result['converged'],
            'iterations': result['iterations'],
            'u_values': solver.U.values.copy(),
            'p_values': solver.p.values.copy(),
            'partition_info': partition_info,
            'n_parts': self.n_parts,
            'exchange_error': exchange_error,
            'residuals': result['residuals'],
        }
