"""
CFL 기반 적응 시간 간격 제어.

Courant-Friedrichs-Lewy 조건 및 Fourier 수 기반 자동 dt 조절.
"""

import numpy as np


class AdaptiveTimeControl:
    """CFL 기반 적응 시간 간격 제어기."""

    def __init__(self, dt_init, dt_min=1e-8, dt_max=1.0,
                 cfl_target=0.5, cfl_max=1.0,
                 fourier_max=0.5,
                 growth_factor=1.2, shrink_factor=0.5,
                 safety_factor=0.9):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.cfl_target = cfl_target
        self.cfl_max = cfl_max
        self.fourier_max = fourier_max
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.safety_factor = safety_factor

        # History tracking
        self.dt_history = []
        self.cfl_history = []
        self.fourier_history = []

    def compute_cfl(self, mesh, velocity, dt):
        """
        CFL number per cell: CFL_i = |u_i| * dt / dx_i

        Parameters
        ----------
        mesh : FVMesh
        velocity : np.ndarray (n_cells,) or (n_cells, ndim)
            Cell-center velocities
        dt : float

        Returns
        -------
        cfl_max : float
        cfl_array : np.ndarray (n_cells,)
        """
        ndim = getattr(mesh, 'ndim', 2)
        n = mesh.n_cells

        # Compute characteristic cell size: dx = V^(1/ndim)
        dx = np.array([mesh.cells[i].volume ** (1.0 / ndim) for i in range(n)])
        dx = np.maximum(dx, 1e-30)

        # Velocity magnitude
        velocity = np.asarray(velocity)
        if velocity.ndim == 1:
            u_mag = np.abs(velocity)
        else:
            u_mag = np.linalg.norm(velocity, axis=1)

        cfl_array = u_mag * dt / dx
        return float(np.max(cfl_array)), cfl_array

    def compute_fourier(self, mesh, alpha_diff, dt):
        """
        Fourier number per cell: Fo_i = alpha * dt / dx_i^2

        Parameters
        ----------
        alpha_diff : float or np.ndarray
            Thermal/momentum diffusivity

        Returns
        -------
        fo_max : float
        fo_array : np.ndarray
        """
        ndim = getattr(mesh, 'ndim', 2)
        n = mesh.n_cells
        dx = np.array([mesh.cells[i].volume ** (1.0 / ndim) for i in range(n)])
        dx = np.maximum(dx, 1e-30)

        fo_array = alpha_diff * dt / (dx ** 2)
        return float(np.max(fo_array)), fo_array

    def compute_dt(self, mesh, velocity, alpha_diff=None, converged=True):
        """
        Compute next time step based on CFL and Fourier constraints.

        Returns
        -------
        dt_new : float
        info : dict with 'cfl_max', 'fourier_max', 'dt_limited_by'
        """
        ndim = getattr(mesh, 'ndim', 2)
        n = mesh.n_cells

        dx = np.array([mesh.cells[i].volume ** (1.0 / ndim) for i in range(n)])
        dx = np.maximum(dx, 1e-30)

        # Velocity magnitude
        velocity = np.asarray(velocity)
        if velocity.ndim == 1:
            u_mag = np.abs(velocity)
        else:
            u_mag = np.linalg.norm(velocity, axis=1)

        # CFL constraint: dt_cfl = cfl_target * dx / |u|
        u_safe = np.maximum(u_mag, 1e-30)
        dt_cfl_arr = self.cfl_target * dx / u_safe
        dt_cfl = float(np.min(dt_cfl_arr)) * self.safety_factor

        dt_new = dt_cfl
        limited_by = 'cfl'

        # Fourier constraint
        if alpha_diff is not None:
            dt_fo_arr = self.fourier_max * dx**2 / np.maximum(alpha_diff, 1e-30)
            dt_fo = float(np.min(dt_fo_arr)) * self.safety_factor
            if dt_fo < dt_new:
                dt_new = dt_fo
                limited_by = 'fourier'

        # Growth limit
        dt_new = min(dt_new, self.growth_factor * self.dt)

        # Shrink on divergence
        if not converged:
            dt_new = self.shrink_factor * self.dt
            limited_by = 'divergence'

        # Clip to bounds
        dt_new = float(np.clip(dt_new, self.dt_min, self.dt_max))

        # Compute actual CFL at new dt
        cfl_max, _ = self.compute_cfl(mesh, velocity, dt_new)
        fo_max = 0.0
        if alpha_diff is not None:
            fo_max, _ = self.compute_fourier(mesh, alpha_diff, dt_new)

        # Store
        self.dt = dt_new
        self.dt_history.append(dt_new)
        self.cfl_history.append(cfl_max)
        self.fourier_history.append(fo_max)

        info = {
            'cfl_max': cfl_max,
            'fourier_max': fo_max,
            'dt_limited_by': limited_by,
            'dt': dt_new
        }
        return dt_new, info

    def get_info(self):
        """Return current state."""
        return {
            'dt': self.dt,
            'dt_history': self.dt_history.copy(),
            'cfl_history': self.cfl_history.copy(),
            'fourier_history': self.fourier_history.copy()
        }
