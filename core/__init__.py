"""FVM 핵심 모듈 — 필드, 연산자, 솔버, 전처리기, GPU 가속."""

from core.fields import ScalarField, VectorField
from core.fvm_operators import (
    FVMSystem, diffusion_operator, source_term,
    convection_operator_upwind, temporal_operator,
    apply_boundary_conditions, under_relax,
)
from core.gradient import green_gauss_gradient
from core.linear_solver import solve_linear_system, solve_pressure_correction
from core.interpolation import compute_mass_flux, muscl_deferred_correction
from core.preconditioner import create_preconditioner
from core.time_control import AdaptiveTimeControl
from core.gpu_solver import detect_gpu_backend

__all__ = [
    "ScalarField", "VectorField",
    "FVMSystem", "diffusion_operator", "source_term",
    "convection_operator_upwind", "temporal_operator",
    "apply_boundary_conditions", "under_relax",
    "green_gauss_gradient",
    "solve_linear_system", "solve_pressure_correction",
    "compute_mass_flux", "muscl_deferred_correction",
    "create_preconditioner",
    "AdaptiveTimeControl",
    "detect_gpu_backend",
]
