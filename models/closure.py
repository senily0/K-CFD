"""
상간 전달 폐합 관계식 (Two-Fluid Model).

- Schiller-Naumann 항력 모델
- Ranz-Marshall 열전달 모델
- Sato 기포유도 난류 모델
- 가상질량력
"""

import numpy as np


def schiller_naumann_drag(Re_p: np.ndarray) -> np.ndarray:
    """
    Schiller-Naumann 항력 계수.

    C_D = 24/Re_p · (1 + 0.15·Re_p^0.687)  (Re_p < 1000)
    C_D = 0.44                                (Re_p >= 1000)

    Parameters
    ----------
    Re_p : 입자 Reynolds 수 배열

    Returns
    -------
    C_D : 항력 계수 배열
    """
    Re_p = np.maximum(Re_p, 1e-10)
    C_D = np.where(
        Re_p < 1000,
        24.0 / Re_p * (1.0 + 0.15 * Re_p**0.687),
        0.44
    )
    return C_D


def drag_force(alpha_g: np.ndarray, rho_l: float, u_g: np.ndarray,
               u_l: np.ndarray, d_b: float, mu_l: float) -> np.ndarray:
    """
    상간 항력.

    F_drag = 0.75·C_D·α_g·ρ_l·|u_g - u_l|·(u_g - u_l) / d_b

    Parameters
    ----------
    alpha_g : 기체 체적분율 (n_cells,)
    rho_l : 액체 밀도
    u_g : 기체 속도 (n_cells, 2)
    u_l : 액체 속도 (n_cells, 2)
    d_b : 기포 직경
    mu_l : 액체 점성

    Returns
    -------
    F_drag : (n_cells, 2) 항력 벡터 (액체에 작용하는 힘)
    """
    u_rel = u_g - u_l  # (n_cells, 2)
    u_rel_mag = np.sqrt(np.sum(u_rel**2, axis=1))
    u_rel_mag = np.maximum(u_rel_mag, 1e-15)

    Re_p = rho_l * u_rel_mag * d_b / mu_l
    C_D = schiller_naumann_drag(Re_p)

    # K_drag = 0.75 * C_D * alpha_g * rho_l * |u_rel| / d_b
    K_drag = 0.75 * C_D * alpha_g * rho_l * u_rel_mag / d_b

    F_drag = K_drag[:, np.newaxis] * u_rel

    return F_drag


def drag_coefficient_implicit(alpha_g: np.ndarray, rho_l: float,
                               u_g: np.ndarray, u_l: np.ndarray,
                               d_b: float, mu_l: float) -> np.ndarray:
    """
    암시적 항력 계수 K_drag (운동량 방정식 커플링용).

    K_drag · (u_g - u_l)로 항력을 표현할 때의 K_drag.

    Returns
    -------
    K_drag : (n_cells,) 항력 계수
    """
    u_rel = u_g - u_l
    u_rel_mag = np.sqrt(np.sum(u_rel**2, axis=1))
    u_rel_mag = np.maximum(u_rel_mag, 1e-15)

    Re_p = rho_l * u_rel_mag * d_b / mu_l
    C_D = schiller_naumann_drag(Re_p)

    K_drag = 0.75 * C_D * alpha_g * rho_l * u_rel_mag / d_b
    return K_drag


def ranz_marshall_nusselt(Re_p: np.ndarray, Pr: float) -> np.ndarray:
    """
    Ranz-Marshall 열전달 상관식.

    Nu = 2 + 0.6·Re_p^0.5·Pr^0.33

    Parameters
    ----------
    Re_p : 입자 Reynolds 수
    Pr : Prandtl 수

    Returns
    -------
    Nu : Nusselt 수
    """
    Re_p = np.maximum(Re_p, 1e-10)
    Nu = 2.0 + 0.6 * Re_p**0.5 * Pr**0.333
    return Nu


def interfacial_heat_transfer(alpha_g: np.ndarray, rho_l: float,
                               u_g: np.ndarray, u_l: np.ndarray,
                               T_g: np.ndarray, T_l: np.ndarray,
                               d_b: float, mu_l: float,
                               cp_l: float, k_l: float) -> np.ndarray:
    """
    상간 열전달률.

    Q = h_i · a_i · (T_g - T_l)
    a_i = 6·α_g / d_b (비계면적)
    h_i = Nu · k_l / d_b

    Returns
    -------
    Q : (n_cells,) 체적당 열전달률 [W/m³]
    """
    u_rel = u_g - u_l
    u_rel_mag = np.sqrt(np.sum(u_rel**2, axis=1))
    u_rel_mag = np.maximum(u_rel_mag, 1e-15)

    Re_p = rho_l * u_rel_mag * d_b / mu_l
    Pr = mu_l * cp_l / k_l
    Nu = ranz_marshall_nusselt(Re_p, Pr)

    h_i = Nu * k_l / d_b
    a_i = 6.0 * alpha_g / d_b

    Q = h_i * a_i * (T_g - T_l)
    return Q


def sato_bubble_induced_turbulence(alpha_g: np.ndarray, rho_l: float,
                                     u_g: np.ndarray, u_l: np.ndarray,
                                     d_b: float) -> np.ndarray:
    """
    Sato 기포유도 난류 점성.

    μ_t,BIT = C_μb · ρ_l · α_g · d_b · |u_g - u_l|
    C_μb = 0.6

    Returns
    -------
    mu_t_BIT : (n_cells,) 기포유도 난류 점성
    """
    C_MU_B = 0.6
    u_rel = u_g - u_l
    u_rel_mag = np.sqrt(np.sum(u_rel**2, axis=1))

    mu_t_BIT = C_MU_B * rho_l * alpha_g * d_b * u_rel_mag
    return mu_t_BIT


def virtual_mass_force(alpha_g: np.ndarray, rho_l: float,
                        du_g_dt: np.ndarray, du_l_dt: np.ndarray,
                        C_vm: float = 0.5) -> np.ndarray:
    """
    가상질량력.

    F_vm = C_vm · α_g · ρ_l · (Du_l/Dt - Du_g/Dt)

    Parameters
    ----------
    du_g_dt, du_l_dt : (n_cells, 2) 가속도
    C_vm : 가상질량 계수 (구: 0.5)

    Returns
    -------
    F_vm : (n_cells, 2) 가상질량력 (기체상에 작용)
    """
    F_vm = C_vm * alpha_g[:, np.newaxis] * rho_l * (du_l_dt - du_g_dt)
    return F_vm
