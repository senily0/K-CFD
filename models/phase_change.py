"""
상변화 모델 모음.

1. LeePhaseChangeModel      — Lee(1980) 체적 증발/응축
2. RohsenowBoilingModel      — Rohsenow(1952) 핵비등 벽면 열유속
3. ZuberCHFModel             — Zuber(1959) 임계 열유속(CHF)
4. NusseltCondensationModel  — Nusselt(1916) 수직 평판 막응축
5. PhaseChangeManager        — 복수 모델 통합 관리

유틸리티:
  saturation_temperature(P) — 간이 Antoine 근사 포화온도
  water_properties(P)       — 압력별 물 물성치 근사
"""

import warnings
import numpy as np
from mesh.mesh_reader import FVMesh
from core.fields import ScalarField


# ============================================================
# 유틸리티: 포화온도 및 물성치 근사
# ============================================================

def saturation_temperature(P_Pa: float) -> float:
    """
    간이 Antoine 근사식으로 물의 포화온도 계산.

    정확도: 0.1~22 MPa 범위에서 IAPWS-IF97 대비 오차 < 2%.

    Parameters
    ----------
    P_Pa : float — 압력 [Pa]

    Returns
    -------
    T_sat : float — 포화온도 [K]
    """
    P_bar = P_Pa / 1e5
    if P_bar < 0.01:
        P_bar = 0.01
    # Antoine 근사 (물, 1~220 bar 범위)
    # log10(P_bar) = A - B / (T_C + C), T_C in Celsius
    # A=5.0768, B=1659.793, C=227.1 (modified for wide range)
    # 역변환: T_C = B / (A - log10(P_bar)) - C
    A, B, C = 5.0768, 1659.793, 227.1
    log_p = np.log10(P_bar)
    denom = A - log_p
    if denom <= 0:
        return 647.0  # 임계온도 근처
    T_C = B / denom - C
    T_K = T_C + 273.15
    return float(np.clip(T_K, 273.15, 647.0))


def water_latent_heat(P_Pa: float) -> float:
    """
    물의 증발 잠열 근사 [J/kg].

    임계점(22.06 MPa)에서 0으로 감소하는 선형 근사.

    Parameters
    ----------
    P_Pa : float — 압력 [Pa]

    Returns
    -------
    h_fg : float — 잠열 [J/kg]
    """
    P_MPa = P_Pa / 1e6
    P_crit = 22.064  # MPa
    if P_MPa >= P_crit:
        return 0.0
    # 1 atm에서 2.257e6, 임계점에서 0, 대략 선형
    h_fg = 2.257e6 * (1.0 - P_MPa / P_crit) ** 0.38
    return float(max(h_fg, 0.0))


def water_properties(P_Pa: float) -> dict:
    """
    압력별 물(액체/증기) 물성치 간이 근사.

    Parameters
    ----------
    P_Pa : float — 압력 [Pa]

    Returns
    -------
    dict with keys: T_sat, h_fg, rho_l, rho_g, cp_l, mu_l, k_l
    """
    T_sat = saturation_temperature(P_Pa)
    h_fg = water_latent_heat(P_Pa)
    P_MPa = P_Pa / 1e6

    # 액체 밀도 근사 (약한 압력 의존성)
    rho_l = 1000.0 - 0.5 * (T_sat - 373.15)
    rho_l = float(np.clip(rho_l, 400.0, 1050.0))

    # 증기 밀도: 이상기체 근사 + 보정
    R_steam = 461.5  # J/(kg·K)
    rho_g = P_Pa / (R_steam * T_sat) if T_sat > 0 else 1.0
    rho_g = float(np.clip(rho_g, 0.01, 500.0))

    return {
        'T_sat': T_sat,
        'h_fg': h_fg,
        'rho_l': rho_l,
        'rho_g': rho_g,
        'cp_l': 4200.0,
        'mu_l': 2.8e-4,
        'k_l': 0.68,
    }


# ============================================================
# 1. Lee 모델 (기존)
# ============================================================

class LeePhaseChangeModel:
    """
    Lee 모델 상변화.

    Parameters
    ----------
    mesh : FVMesh
    T_sat : float - 포화 온도 [K]
    r_evap : float - 증발 계수 [1/s] (기본 0.1)
    r_cond : float - 응축 계수 [1/s] (기본 0.1)
    L_latent : float - 잠열 [J/kg] (기본 2.26e6, 물)
    rho_l : float - 액체 밀도
    rho_g : float - 기체 밀도
    """

    def __init__(self, mesh: FVMesh, T_sat: float = 373.15,
                 r_evap: float = 0.1, r_cond: float = 0.1,
                 L_latent: float = 2.26e6,
                 rho_l: float = 1000.0, rho_g: float = 1.0):
        self.mesh = mesh
        self.T_sat = T_sat
        self.r_evap = r_evap
        self.r_cond = r_cond
        self.L_latent = L_latent
        self.rho_l = rho_l
        self.rho_g = rho_g

    def compute_mass_transfer(self, T: ScalarField, alpha_l: ScalarField) -> np.ndarray:
        """
        질량 전환율 계산.

        Returns
        -------
        dot_m : (n_cells,) 질량 전환율 [kg/(m3*s)]
                양수: 증발 (액체->기체), 음수: 응축 (기체->액체)
        """
        n = self.mesh.n_cells
        dot_m = np.zeros(n)
        T_val = T.values
        al = alpha_l.values
        ag = 1.0 - al

        mask_evap = T_val > self.T_sat
        mask_cond = T_val < self.T_sat

        dot_m[mask_evap] = (self.r_evap * self.rho_l * al[mask_evap]
                            * (T_val[mask_evap] - self.T_sat) / self.T_sat)
        dot_m[mask_cond] = (-self.r_cond * self.rho_g * ag[mask_cond]
                            * (self.T_sat - T_val[mask_cond]) / self.T_sat)
        return dot_m

    def get_source_terms(self, T: ScalarField, alpha_l: ScalarField) -> dict:
        """
        상변화에 의한 소스항 반환.

        Returns
        -------
        sources : dict with keys:
            'alpha_l' : (n,) 액체 체적분율 소스
            'alpha_g' : (n,) 기체 체적분율 소스
            'energy'  : (n,) 에너지 소스 [W/m3]
        """
        dot_m = self.compute_mass_transfer(T, alpha_l)

        sources = {
            'alpha_l': -dot_m / self.rho_l,    # 증발 시 액체 감소
            'alpha_g':  dot_m / self.rho_g,    # 증발 시 기체 증가
            'energy':  -dot_m * self.L_latent, # 증발 시 에너지 흡수
        }
        return sources


# ============================================================
# 2. Rohsenow 핵비등 모델
# ============================================================

class RohsenowBoilingModel:
    """
    Rohsenow(1952) 핵비등 벽면 열유속 상관식.

    q''_w = mu_l * h_fg * [g*(rho_l - rho_g)/sigma]^0.5
            * [cp_l * dT_sat / (C_sf * h_fg * Pr_l^n)]^3

    Parameters
    ----------
    T_sat : float   — 포화 온도 [K]
    h_fg  : float   — 잠열 [J/kg]
    rho_l : float   — 액체 밀도 [kg/m3]
    rho_g : float   — 기체 밀도 [kg/m3]
    mu_l  : float   — 액체 점도 [Pa·s]
    cp_l  : float   — 액체 비열 [J/(kg·K)]
    sigma : float   — 표면장력 [N/m]
    Pr_l  : float   — 액체 Prandtl 수
    C_sf  : float   — 표면-유체 상수 (기본 0.013, 물-구리)
    n     : float   — Prandtl 지수 (기본 1.0, 물)
    g     : float   — 중력가속도 [m/s2]
    """

    def __init__(self, T_sat: float, h_fg: float,
                 rho_l: float, rho_g: float,
                 mu_l: float, cp_l: float,
                 sigma: float, Pr_l: float,
                 C_sf: float = 0.013, n: float = 1.0,
                 g: float = 9.81):
        self.T_sat = T_sat
        self.h_fg = h_fg
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.cp_l = cp_l
        self.sigma = sigma
        self.Pr_l = Pr_l
        self.C_sf = C_sf
        self.n = n
        self.g = g

    def compute_wall_heat_flux(self, T_wall: float) -> float:
        """
        벽면 열유속 [W/m2] 계산.

        Parameters
        ----------
        T_wall : float — 벽면 온도 [K] (T_wall > T_sat 일 때 비등)

        Returns
        -------
        q_w : float — 벽면 열유속 [W/m2] (양수 = 벽→유체)
        """
        dT = T_wall - self.T_sat
        if dT <= 0.0:
            return 0.0

        term1 = self.mu_l * self.h_fg
        term2 = np.sqrt(self.g * (self.rho_l - self.rho_g) / self.sigma)
        term3 = (self.cp_l * dT / (self.C_sf * self.h_fg * self.Pr_l ** self.n)) ** 3

        return float(term1 * term2 * term3)

    def compute_mass_transfer_wall(self, T_wall: float,
                                   A_wall: float, V_cell: float) -> float:
        """
        벽면 비등에 의한 질량 전환율 [kg/(m3·s)].

        Parameters
        ----------
        T_wall : float — 벽면 온도 [K]
        A_wall : float — 벽면 면적 [m2]
        V_cell : float — 셀 체적 [m3]

        Returns
        -------
        dot_m : float — 질량 전환율 [kg/(m3·s)], 양수=증발
        """
        q = self.compute_wall_heat_flux(T_wall)
        if V_cell <= 0.0 or self.h_fg <= 0.0:
            return 0.0
        return q * A_wall / (self.h_fg * V_cell)


# ============================================================
# 3. Zuber 임계 열유속 (CHF) 모델
# ============================================================

class ZuberCHFModel:
    """
    Zuber(1959) 풀비등 임계 열유속(CHF) 상관식.

    q''_CHF = 0.131 * rho_g * h_fg * [sigma * g * (rho_l - rho_g) / rho_g^2]^0.25

    Parameters
    ----------
    h_fg  : float — 잠열 [J/kg]
    rho_l : float — 액체 밀도 [kg/m3]
    rho_g : float — 기체 밀도 [kg/m3]
    sigma : float — 표면장력 [N/m]
    g     : float — 중력가속도 [m/s2]
    """

    def __init__(self, h_fg: float, rho_l: float, rho_g: float,
                 sigma: float, g: float = 9.81):
        self.h_fg = h_fg
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.sigma = sigma
        self.g = g

    def compute_chf(self) -> float:
        """
        임계 열유속 [W/m2] 계산.
        """
        term = (self.sigma * self.g * (self.rho_l - self.rho_g)
                / self.rho_g ** 2) ** 0.25
        return float(0.131 * self.rho_g * self.h_fg * term)

    def check_margin(self, q_wall: float) -> dict:
        """
        열유속 대비 CHF 여유도 확인.

        Returns
        -------
        dict with 'chf', 'ratio' (q_wall/chf), 'safe' (ratio < 1)
        """
        chf = self.compute_chf()
        ratio = q_wall / chf if chf > 0 else float('inf')
        if ratio >= 1.0:
            warnings.warn(
                f"벽면 열유속({q_wall:.2e} W/m2)이 "
                f"CHF({chf:.2e} W/m2)에 도달하였습니다.",
                RuntimeWarning, stacklevel=2)
        return {'chf': chf, 'ratio': ratio, 'safe': ratio < 1.0}


# ============================================================
# 4. Nusselt 막응축 모델
# ============================================================

class NusseltCondensationModel:
    """
    Nusselt(1916) 수직 평판 막응축 해석해.

    h_cond = 0.943 * [rho_l*(rho_l - rho_g)*g*h_fg*k_l^3
                       / (mu_l * L * dT_sub)]^0.25

    Parameters
    ----------
    T_sat : float — 포화 온도 [K]
    h_fg  : float — 잠열 [J/kg]
    rho_l : float — 액체 밀도 [kg/m3]
    rho_g : float — 기체 밀도 [kg/m3]
    mu_l  : float — 액체 점도 [Pa·s]
    k_l   : float — 액체 열전도도 [W/(m·K)]
    g     : float — 중력가속도 [m/s2]
    """

    def __init__(self, T_sat: float, h_fg: float,
                 rho_l: float, rho_g: float,
                 mu_l: float, k_l: float,
                 g: float = 9.81):
        self.T_sat = T_sat
        self.h_fg = h_fg
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.k_l = k_l
        self.g = g

    def compute_heat_transfer_coeff(self, L_plate: float,
                                    delta_T_sub: float) -> float:
        """
        Nusselt 막응축 열전달 계수 [W/(m2·K)].

        Parameters
        ----------
        L_plate    : float — 평판 높이 [m]
        delta_T_sub: float — 과냉도 = T_sat - T_wall [K]

        Returns
        -------
        h_cond : float — 평균 열전달 계수 [W/(m2·K)]
        """
        if delta_T_sub <= 0.0 or L_plate <= 0.0:
            return 0.0

        numer = (self.rho_l * (self.rho_l - self.rho_g) * self.g
                 * self.h_fg * self.k_l ** 3)
        denom = self.mu_l * L_plate * delta_T_sub

        if denom <= 0.0:
            return 0.0

        return float(0.943 * (numer / denom) ** 0.25)

    def compute_condensation_rate(self, L_plate: float, T_wall: float,
                                  A_wall: float, V_cell: float) -> float:
        """
        응축에 의한 질량 전환율 [kg/(m3·s)].

        Parameters
        ----------
        L_plate : float — 평판 높이 [m]
        T_wall  : float — 벽면 온도 [K] (T_wall < T_sat)
        A_wall  : float — 벽면 면적 [m2]
        V_cell  : float — 셀 체적 [m3]

        Returns
        -------
        dot_m : float — 질량 전환율 [kg/(m3·s)], 음수=응축(기체→액체)
        """
        dT_sub = self.T_sat - T_wall
        if dT_sub <= 0.0:
            return 0.0

        h = self.compute_heat_transfer_coeff(L_plate, dT_sub)
        q = h * dT_sub  # 열유속 [W/m2]

        if V_cell <= 0.0 or self.h_fg <= 0.0:
            return 0.0

        # 응축 = 기체→액체, 음수로 반환
        return -q * A_wall / (self.h_fg * V_cell)


# ============================================================
# 5. PhaseChangeManager — 통합 관리
# ============================================================

class PhaseChangeManager:
    """
    복수 상변화 모델 통합 관리 인터페이스.

    Lee 모델(체적 상변화)을 기본으로 사용하고,
    선택적으로 Rohsenow 비등/Nusselt 응축 벽면 모델을 추가한다.

    Parameters
    ----------
    mesh    : FVMesh
    T_sat   : float — 포화 온도 [K]
    h_fg    : float — 잠열 [J/kg]
    rho_l   : float — 액체 밀도 [kg/m3]
    rho_g   : float — 기체 밀도 [kg/m3]
    lee_params      : dict  — LeePhaseChangeModel 추가 파라미터
    boiling_params  : dict  — RohsenowBoilingModel 파라미터 (None이면 비활성)
    condensation_params : dict — NusseltCondensationModel 파라미터 (None이면 비활성)
    """

    def __init__(self, mesh: FVMesh, T_sat: float = 373.15,
                 h_fg: float = 2.26e6,
                 rho_l: float = 1000.0, rho_g: float = 1.0,
                 lee_params: dict = None,
                 boiling_params: dict = None,
                 condensation_params: dict = None):
        self.mesh = mesh
        self.T_sat = T_sat
        self.h_fg = h_fg
        self.rho_l = rho_l
        self.rho_g = rho_g

        # Lee 모델 (항상 활성)
        lp = lee_params or {}
        self.lee_model = LeePhaseChangeModel(
            mesh, T_sat=T_sat,
            L_latent=h_fg, rho_l=rho_l, rho_g=rho_g,
            **lp)

        # Rohsenow 비등 모델 (선택)
        self.boiling_model = None
        if boiling_params is not None:
            self.boiling_model = RohsenowBoilingModel(
                T_sat=T_sat, h_fg=h_fg,
                rho_l=rho_l, rho_g=rho_g,
                **boiling_params)

        # Nusselt 응축 모델 (선택)
        self.condensation_model = None
        if condensation_params is not None:
            self.condensation_model = NusseltCondensationModel(
                T_sat=T_sat, h_fg=h_fg,
                rho_l=rho_l, rho_g=rho_g,
                **condensation_params)

    def compute_total_mass_transfer(self, T: ScalarField,
                                    alpha_l: ScalarField,
                                    wall_cells: list = None,
                                    T_wall: float = None,
                                    A_wall: float = None,
                                    V_cell: float = None,
                                    L_plate: float = None) -> np.ndarray:
        """
        모든 활성 모델의 질량 전환율 합산.

        Parameters
        ----------
        T, alpha_l : ScalarField — 온도/체적분율 필드
        wall_cells : list[int]   — 벽면 인접 셀 인덱스
        T_wall     : float       — 벽면 온도 [K]
        A_wall     : float       — 셀당 벽면 면적 [m2]
        V_cell     : float       — 셀 체적 [m3]
        L_plate    : float       — 응축 평판 높이 [m]

        Returns
        -------
        dot_m : (n_cells,) 총 질량 전환율 [kg/(m3·s)]
        """
        # 1) Lee 체적 상변화
        dot_m = self.lee_model.compute_mass_transfer(T, alpha_l)

        # 2) 벽면 비등 (Rohsenow)
        if (self.boiling_model is not None and wall_cells is not None
                and T_wall is not None and T_wall > self.T_sat):
            for ci in wall_cells:
                a_w = A_wall if A_wall else 1.0
                v_c = V_cell if V_cell else 1.0
                dm = self.boiling_model.compute_mass_transfer_wall(
                    T_wall, a_w, v_c)
                dot_m[ci] += dm

        # 3) 벽면 응축 (Nusselt)
        if (self.condensation_model is not None and wall_cells is not None
                and T_wall is not None and T_wall < self.T_sat):
            Lp = L_plate if L_plate else 0.1
            for ci in wall_cells:
                a_w = A_wall if A_wall else 1.0
                v_c = V_cell if V_cell else 1.0
                dm = self.condensation_model.compute_condensation_rate(
                    Lp, T_wall, a_w, v_c)
                dot_m[ci] += dm

        return dot_m

    def get_source_terms(self, T: ScalarField, alpha_l: ScalarField,
                         **kwargs) -> dict:
        """
        alpha, energy 소스항 반환.

        Returns
        -------
        sources : dict with 'alpha_l', 'alpha_g', 'energy'
        """
        dot_m = self.compute_total_mass_transfer(T, alpha_l, **kwargs)

        return {
            'alpha_l': -dot_m / self.rho_l,
            'alpha_g':  dot_m / self.rho_g,
            'energy':  -dot_m * self.h_fg,
        }
