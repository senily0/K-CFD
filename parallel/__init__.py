"""MPI 병렬화 모듈 — 영역 분할, 병렬 솔버."""

from parallel.partitioning import GeometricPartitioner, GhostCellLayer
from parallel.mpi_solver import MPISIMPLESolver

__all__ = [
    "GeometricPartitioner", "GhostCellLayer",
    "MPISIMPLESolver",
]
