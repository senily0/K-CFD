"""격자 생성, 읽기, AMR, VTK 내보내기 모듈."""

from mesh.mesh_reader import FVMesh, build_fvmesh_from_arrays
from mesh.mesh_generator import (
    generate_channel_mesh, generate_cavity_mesh,
    generate_cht_mesh, generate_bubble_column_mesh,
    generate_triangle_channel_mesh, _make_structured_quad_mesh,
)
from mesh.mesh_generator_3d import (
    generate_3d_duct_mesh, generate_3d_cavity_mesh,
    generate_3d_channel_mesh,
)
from mesh.hybrid_mesh_generator import generate_hybrid_hex_tet_mesh
from mesh.vtk_exporter import export_mesh_to_vtu, export_input_json

# mesh.amr imports from core.fields, creating a cross-package cycle when both
# packages are initialised eagerly. Import lazily via mesh.amr directly.
def __getattr__(name):
    if name in ("AMRMesh", "GradientJumpEstimator", "AMRSolverLoop"):
        from mesh import amr as _amr
        return getattr(_amr, name)
    raise AttributeError(f"module 'mesh' has no attribute {name!r}")

__all__ = [
    "FVMesh", "build_fvmesh_from_arrays",
    "generate_channel_mesh", "generate_cavity_mesh",
    "generate_cht_mesh", "generate_bubble_column_mesh",
    "generate_triangle_channel_mesh", "_make_structured_quad_mesh",
    "generate_3d_duct_mesh", "generate_3d_cavity_mesh",
    "generate_3d_channel_mesh",
    "generate_hybrid_hex_tet_mesh",
    "export_mesh_to_vtu", "export_input_json",
    "AMRMesh", "GradientJumpEstimator", "AMRSolverLoop",
]
