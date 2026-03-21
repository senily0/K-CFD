# Case 10: Chemical Reaction Transport

## Problem Description

One-dimensional advection-diffusion-reaction of a scalar species with a first-order
irreversible reaction A -> B. A uniform flow carries species A through a channel
where it decays exponentially. This verifies the scalar transport equation with a
source term and the coupling between convection, diffusion, and reaction.

## Geometry

```
  insulated (symmetry)
  ================================================
  C_A(inlet)=1  -->  -->  -->  -->  C_A(x)  -->
  inlet                                   outlet
  -->  -->  -->  -->  -->  -->  -->  -->  -->
  ================================================
  insulated (symmetry)

  Lx = 1.0 m, Ly = 0.02 m (quasi-1D)
```

## Boundary Conditions

| Boundary | Type | Value |
|----------|------|-------|
| Inlet | Dirichlet | C_A = 1.0 mol/m^3, u = 1.0 m/s |
| Outlet | Zero gradient | dC_A/dx = 0 |
| Top / Bottom | Symmetry | dC_A/dy = 0 |

## Physical Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Inlet velocity (u) | 1.0 | m/s |
| Diffusivity (D) | 0.001 | m^2/s |
| Reaction rate (k_r) | 1.0 | 1/s |
| Inlet concentration (C_0) | 1.0 | mol/m^3 |
| Peclet number (Pe) | 1000 | - |
| Damkohler number (Da) | 1.0 | - |

## Analytical Solution

For a 1D steady-state advection-diffusion-reaction equation with first-order
kinetics:

```
u * dC/dx = D * d^2C/dx^2 - k_r * C

C(x) = C_0 * exp(lambda * x)
lambda = (u - sqrt(u^2 + 4*D*k_r)) / (2*D)
```

For Pe >> 1 (convection-dominated), the solution simplifies to:

```
C(x) ~ C_0 * exp(-k_r * x / u)
```

## Verification Method

This case is verified internally by `verification_cases.exe` (Case 10).
The CLI solver does not support species transport with reaction source terms.

## Expected Results

- Concentration profile matches analytical exponential decay
- L2 concentration error < 5%
- Mass balance: integral of reaction rate equals inlet - outlet flux
- Damkohler number effect correctly captured

## How to Run

```bash
# Via verification_cases.exe (internal test)
cd cpp/build
./verification_cases.exe    # Runs all cases including Case 10

# Via Python verification suite
cd VV
python run_all.py --case 10
```

## Reference

- Bird, R.B., Stewart, W.E. & Lightfoot, E.N., *Transport Phenomena*,
  2nd ed., Wiley (2002), Chapter 18.
- Levenspiel, O., *Chemical Reaction Engineering*, 3rd ed., Wiley (1999).
