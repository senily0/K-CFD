"""
단상 비압축성 Navier-Stokes 솔버 (SIMPLE 알고리즘).

2D 비정렬 FVM 기반, 셀 중심법.
면 법선은 owner→neighbour 방향 (경계면은 외향).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField, VectorField
from core.gradient import green_gauss_gradient


class SIMPLESolver:
    """SIMPLE 알고리즘 기반 비압축성 NS 솔버."""

    def __init__(self, mesh: FVMesh, rho: float = 1.0, mu: float = 1e-3):
        self.mesh = mesh
        self.rho = rho
        self.mu = mu

        self.U = VectorField(mesh, "velocity")
        self.p = ScalarField(mesh, "pressure")

        self.alpha_u = 0.7
        self.alpha_p = 0.3
        self.max_outer_iter = 500
        self.tol = 1e-6

        self.bc_u: dict = {}
        self.bc_v: dict = {}
        self.bc_p: dict = {}
        self.residuals: list = []
        self.turbulence_model = None
        self.transient = False
        self.dt = 0.01
        self._res0 = None  # initial residual for normalization

        n = mesh.n_cells
        ndim = getattr(mesh, 'ndim', 2)
        self._aP = {comp: np.ones(n) for comp in range(ndim)}

        # 캐시
        self._face_bc_cache: dict = {}
        self._build_face_bc_cache()

    def _build_face_bc_cache(self):
        self._face_bc_cache = {}
        for bname, fids in self.mesh.boundary_patches.items():
            for li, fid in enumerate(fids):
                self._face_bc_cache[fid] = (bname, li)

    def set_velocity_bc(self, patch: str, bc_type: str, value=None):
        if patch not in self.mesh.boundary_patches:
            return
        if value is not None:
            self.U.set_boundary(patch, np.array(value, dtype=float))
        self.bc_u[patch] = {'type': bc_type}
        self.bc_v[patch] = {'type': bc_type}

    def set_pressure_bc(self, patch: str, bc_type: str, value=None):
        if patch not in self.mesh.boundary_patches:
            return
        if value is not None:
            self.p.set_boundary(patch, value)
        self.bc_p[patch] = {'type': bc_type}

    def solve_steady(self) -> dict:
        self.residuals = []
        self._res0 = None
        n = self.mesh.n_cells
        ndim = getattr(self.mesh, 'ndim', 2)
        self._aP = {comp: np.ones(n) for comp in range(ndim)}

        for it in range(self.max_outer_iter):
            mf = self._face_mass_flux()
            res_mom = []
            for comp in range(ndim):
                res_mom.append(self._momentum_eq(comp, mf))
            # recompute mf with updated velocity
            mf = self._face_mass_flux()
            res_p = self._pressure_correction(mf)

            res = max(max(res_mom), res_p)

            # normalize by initial residual
            if self._res0 is None and res > 1e-30:
                self._res0 = res
            res_norm = res / self._res0 if self._res0 and self._res0 > 1e-30 else res
            self.residuals.append(res_norm)

            if it < 5 or it % 100 == 0:
                mom_str = " ".join(f"c{c}={res_mom[c]:.2e}" for c in range(ndim))
                print(f"    iter {it:4d}: res={res_norm:.3e}  (raw: {mom_str} p={res_p:.2e})")
            if res_norm < self.tol:
                return {'converged': True, 'iterations': it + 1,
                        'residuals': self.residuals}
        return {'converged': False, 'iterations': self.max_outer_iter,
                'residuals': self.residuals}

    def solve_transient(self, t_end: float, dt: float = None) -> dict:
        if dt: self.dt = dt
        self.transient = True
        n = self.mesh.n_cells
        ndim = getattr(self.mesh, 'ndim', 2)
        self._aP = {comp: np.ones(n) for comp in range(ndim)}
        t, steps = 0.0, 0
        while t < t_end - 1e-15:
            self.U.store_old(); self.p.store_old()
            for _ in range(30):
                mf = self._face_mass_flux()
                for comp in range(ndim):
                    self._momentum_eq(comp, mf)
                mf = self._face_mass_flux()
                rp = self._pressure_correction(mf)
                if rp < self.tol * 10: break
            t += self.dt; steps += 1
        self.transient = False
        return {'time_steps': steps, 'final_time': t}

    # ---- face mass flux ----
    def _face_mass_flux(self) -> np.ndarray:
        mesh = self.mesh
        mf = np.zeros(mesh.n_faces)
        for fid, face in enumerate(mesh.faces):
            o = face.owner
            if face.neighbour >= 0:
                gc = self._gc(fid)
                uf = gc * self.U.values[o] + (1 - gc) * self.U.values[face.neighbour]
            else:
                info = self._face_bc_cache.get(fid)
                if info:
                    bname, li = info
                    bc = self.bc_u.get(bname, {'type': 'zero_gradient'})
                    if bc['type'] == 'dirichlet':
                        uf = self.U.boundary_values[bname][li]
                    else:
                        uf = self.U.values[o]
                else:
                    uf = self.U.values[o]
            mf[fid] = self.rho * np.dot(uf, face.normal) * face.area
        return mf

    # ---- momentum equation ----
    def _momentum_eq(self, comp: int, mf: np.ndarray) -> float:
        mesh = self.mesh
        n = mesh.n_cells
        mu = self.mu

        mu_t = None
        if self.turbulence_model is not None:
            mu_t = self.turbulence_model.get_turbulent_viscosity().values

        aP = np.zeros(n)
        aN_r, aN_c, aN_v = [], [], []
        b = np.zeros(n)

        for fid, face in enumerate(mesh.faces):
            o = face.owner
            F = mf[fid]

            if face.neighbour >= 0:
                nb = face.neighbour
                d = np.linalg.norm(mesh.cells[nb].center - mesh.cells[o].center)
                if d < 1e-30:
                    continue

                mu_o = mu + (mu_t[o] if mu_t is not None else 0)
                mu_n = mu + (mu_t[nb] if mu_t is not None else 0)
                # Non-orthogonal correction: use orthogonal distance
                d_vec = mesh.cells[nb].center - mesh.cells[o].center
                d_orth = abs(np.dot(d_vec, face.normal))
                d_orth = max(d_orth, 0.1 * d)  # safety limit
                Df = 2 * mu_o * mu_n / max(mu_o + mu_n, 1e-30) * face.area / d_orth

                # Upwind convection + central diffusion
                # owner equation: aP*phi_o + aN*phi_nb = source
                # aP contribution from this face: Df + max(F,0)
                # aN contribution: -(Df + max(-F,0))  [i.e. -(Df - min(F,0))]
                aP[o] += Df + max(F, 0)
                aN_r.append(o); aN_c.append(nb); aN_v.append(-(Df + max(-F, 0)))

                # neighbour equation (flux direction reversed: -F)
                aP[nb] += Df + max(-F, 0)
                aN_r.append(nb); aN_c.append(o); aN_v.append(-(Df + max(F, 0)))

            else:
                info = self._face_bc_cache.get(fid)
                if info is None:
                    continue
                bname, li = info
                bc_dict = self.bc_u  # all velocity components share same BC type
                bc = bc_dict.get(bname, {'type': 'zero_gradient'})

                d = np.linalg.norm(face.center - mesh.cells[o].center)
                if d < 1e-30:
                    continue
                mu_o = mu + (mu_t[o] if mu_t is not None else 0)
                Df = mu_o * face.area / d

                if bc['type'] == 'dirichlet':
                    phi_b = self.U.boundary_values[bname][li, comp]
                    # Diffusion: Df*(phi_b - phi_P) → aP += Df, b += Df*phi_b
                    aP[o] += Df
                    b[o] += Df * phi_b
                    # Convection boundary:
                    if F >= 0:
                        # outflow: uses cell value
                        aP[o] += F
                    else:
                        # inflow: uses boundary value
                        b[o] += (-F) * phi_b
                else:  # zero_gradient
                    # Diffusion: zero (no gradient)
                    # Convection: uses cell value regardless
                    if F >= 0:
                        aP[o] += F
                    else:
                        aP[o] += (-F)

        # time
        if self.transient and self.U.old_values is not None:
            for ci in range(n):
                c = self.rho * mesh.cells[ci].volume / self.dt
                aP[ci] += c
                b[ci] += c * self.U.old_values[ci, comp]

        # pressure gradient source: -∫ p·n_comp dA ≈ -Σ_f p_f·(n·e_comp)·A_f
        for fid, face in enumerate(mesh.faces):
            o = face.owner
            if face.neighbour >= 0:
                nb = face.neighbour
                pf = 0.5 * (self.p.values[o] + self.p.values[nb])
                src = pf * face.normal[comp] * face.area
                b[o] -= src
                b[nb] += src
            else:
                info = self._face_bc_cache.get(fid)
                if info:
                    bname, li = info
                    pbc = self.bc_p.get(bname, {'type': 'zero_gradient'})
                    pf = (self.p.boundary_values[bname][li]
                          if pbc['type'] == 'dirichlet'
                          else self.p.values[o])
                else:
                    pf = self.p.values[o]
                b[o] -= pf * face.normal[comp] * face.area

        # under-relaxation: aP_new = aP/alpha, b_new = b + (1-alpha)/alpha * aP * phi_old
        alpha = self.alpha_u
        phi_old = self.U.values[:, comp].copy()
        b += (1 - alpha) / alpha * aP * phi_old
        aP /= alpha

        # save unrelaxed aP for pressure correction (multiply back by alpha)
        self._aP[comp] = aP * alpha  # the physical aP without relaxation

        # assemble and solve
        rows = aN_r + list(range(n))
        cols = aN_c + list(range(n))
        vals = aN_v + list(aP)
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        phi = spsolve(A, b)
        if np.any(np.isnan(phi)):
            phi = phi_old
        self.U.values[:, comp] = phi

        r = np.linalg.norm(A @ phi - b)
        return r / max(np.linalg.norm(b), 1e-15)

    # ---- pressure correction ----
    def _pressure_correction(self, mf: np.ndarray) -> float:
        mesh = self.mesh
        n = mesh.n_cells

        # d_P = V_P / a_P for each cell (unrelaxed aP)
        ndim = getattr(mesh, 'ndim', 2)
        aP_sum = sum(self._aP[c] for c in range(ndim))
        aP_mom = np.maximum(aP_sum / ndim, 1e-30)
        dP = np.array([mesh.cells[ci].volume for ci in range(n)]) / aP_mom

        rows, cols, vals = [], [], []
        aP_pp = np.zeros(n)
        b = np.zeros(n)

        for fid, face in enumerate(mesh.faces):
            o = face.owner
            if face.neighbour >= 0:
                nb = face.neighbour
                d = np.linalg.norm(mesh.cells[nb].center - mesh.cells[o].center)
                if d < 1e-30:
                    continue

                # Non-orthogonal correction: use orthogonal distance
                d_vec = mesh.cells[nb].center - mesh.cells[o].center
                d_orth = abs(np.dot(d_vec, face.normal))
                d_orth = max(d_orth, 0.1 * d)
                df = self.rho * 0.5 * (dP[o] + dP[nb]) * face.area / d_orth

                aP_pp[o] += df
                aP_pp[nb] += df
                rows.append(o); cols.append(nb); vals.append(-df)
                rows.append(nb); cols.append(o); vals.append(-df)

                b[o] -= mf[fid]
                b[nb] += mf[fid]
            else:
                # boundary mass imbalance
                b[o] -= mf[fid]

                # For Dirichlet pressure boundaries: p' = 0 at boundary
                # Add diffusion-like term: df*(p'_b - p'_P) with p'_b=0
                # → aP += df, b += 0
                info = self._face_bc_cache.get(fid)
                if info:
                    bname, li = info
                    pbc = self.bc_p.get(bname, {'type': 'zero_gradient'})
                    if pbc['type'] == 'dirichlet':
                        d = np.linalg.norm(face.center - mesh.cells[o].center)
                        if d > 1e-30:
                            df = self.rho * dP[o] * face.area / d
                            aP_pp[o] += df
                            # b[o] += df * 0 (p'_boundary = 0)

        # diagonal
        for ci in range(n):
            rows.append(ci); cols.append(ci); vals.append(aP_pp[ci])

        # reference pressure if no Dirichlet p
        has_p_dir = any(v.get('type') == 'dirichlet' for v in self.bc_p.values())
        if not has_p_dir:
            # Fix one cell's pressure correction to zero (row 0)
            # Zero out row 0 off-diagonal entries and set diagonal = 1, b = 0
            # This is cleaner than adding a large penalty
            kill_rows = set()
            kill_rows.add(0)
            new_vals = []
            for i, (r, c, v) in enumerate(zip(rows, cols, vals)):
                if r in kill_rows:
                    new_vals.append(0.0)
                else:
                    new_vals.append(v)
            vals = new_vals
            # set diagonal and RHS for reference cell
            rows.append(0); cols.append(0); vals.append(1.0)
            b[0] = 0.0

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        pp = spsolve(A, b)
        if np.any(np.isnan(pp)):
            pp = np.zeros(n)

        # Velocity correction: u'_P = -d_P * grad(p')_P
        # Compute grad(p') via Green-Gauss
        pp_field = ScalarField(mesh, "pp")
        pp_field.values = pp.copy()
        for bname in mesh.boundary_patches:
            pbc = self.bc_p.get(bname, {'type': 'zero_gradient'})
            if pbc['type'] == 'dirichlet':
                pp_field.set_boundary(bname, 0.0)
            else:
                fids = mesh.boundary_patches[bname]
                for li, fid in enumerate(fids):
                    pp_field.boundary_values[bname][li] = pp[mesh.faces[fid].owner]

        grad_pp = green_gauss_gradient(pp_field)

        for ci in range(n):
            for comp in range(ndim):
                self.U.values[ci, comp] -= dP[ci] * grad_pp[ci, comp]

        # Pressure update
        self.p.values += self.alpha_p * pp

        return float(np.linalg.norm(pp) / max(np.linalg.norm(self.p.values) + 1e-10, 1e-10))

    # ---- helpers ----
    def _gc(self, fid):
        face = self.mesh.faces[fid]
        xO = self.mesh.cells[face.owner].center
        xN = self.mesh.cells[face.neighbour].center
        xF = face.center
        dO = np.linalg.norm(xF - xO)
        dN = np.linalg.norm(xF - xN)
        t = dO + dN
        return dN / t if t > 1e-30 else 0.5

    def get_velocity_at_y(self, x_target, n_points=50):
        mesh = self.mesh
        dx = 0
        for fid, face in enumerate(mesh.faces):
            if face.neighbour >= 0:
                d = abs(mesh.cells[face.owner].center[0] - mesh.cells[face.neighbour].center[0])
                if d > 1e-10: dx = d; break
        if dx == 0: dx = 0.05
        ys, us, vs = [], [], []
        for ci in range(mesh.n_cells):
            if abs(mesh.cells[ci].center[0] - x_target) < dx * 1.5:
                ys.append(mesh.cells[ci].center[1])
                us.append(self.U.values[ci, 0])
                vs.append(self.U.values[ci, 1])
        idx = np.argsort(ys)
        return np.array(ys)[idx], np.array(us)[idx], np.array(vs)[idx]

    def get_velocity_at_x(self, y_target):
        mesh = self.mesh
        dy = 0
        for fid, face in enumerate(mesh.faces):
            if face.neighbour >= 0:
                d = abs(mesh.cells[face.owner].center[1] - mesh.cells[face.neighbour].center[1])
                if d > 1e-10: dy = d; break
        if dy == 0: dy = 0.05
        xs, us, vs = [], [], []
        for ci in range(mesh.n_cells):
            if abs(mesh.cells[ci].center[1] - y_target) < dy * 1.5:
                xs.append(mesh.cells[ci].center[0])
                us.append(self.U.values[ci, 0])
                vs.append(self.U.values[ci, 1])
        idx = np.argsort(xs)
        return np.array(xs)[idx], np.array(us)[idx], np.array(vs)[idx]
