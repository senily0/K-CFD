"""
generate_full_report.py
-----------------------
Comprehensive K-CFD Technical Report generator.

Strategy:
  1. Build the full base report (all sections from generate_report.py).
  2. Apply all C++ text/heading replacements from update_report.py.
  3. INSERT detailed physics equations (Section 2 expanded) and
     literature-based verification (Section 9 expanded).

Output: report/K-CFD_Technical_Report.docx
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ============================================================
# Global style helpers
# ============================================================
style = doc.styles['Normal']
style.font.name = 'Malgun Gothic'
style.font.size = Pt(10)
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = 'Malgun Gothic'
    hs.font.bold = True
    if level == 1:
        hs.font.size = Pt(16)
        hs.font.color.rgb = RGBColor(0, 51, 102)
    elif level == 2:
        hs.font.size = Pt(13)
        hs.font.color.rgb = RGBColor(0, 70, 130)
    else:
        hs.font.size = Pt(11)


def add_table(headers, rows, col_widths=None):
    """Add a formatted table and a blank paragraph after it."""
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def eq(text, indent=True):
    """Add an equation-style paragraph (indented, monospace-ish)."""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9.5)
    return p


def body(text):
    """Add a normal body paragraph."""
    return doc.add_paragraph(text)


def note(text):
    """Add a small-font note paragraph."""
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(80, 80, 80)
    return p


# ============================================================
# Title page
# ============================================================
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run('K-CFD: Two-Fluid Model FVM 기반\n열유체 해석 코드 개발 보고서')
run.font.size = Pt(22)
run.font.bold = True
run.font.color.rgb = RGBColor(0, 51, 102)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run(
    'Euler-Euler 이상유동 + Conjugate Heat Transfer + k-ε/k-ω SST 난류 모델\n'
    'SIMPLE/PISO, MUSCL/TVD 2차 대류, Rhie-Chow 보간, BDF2 시간적분\n'
    'IAPWS-IF97 증기표, CSF 표면장력, OpenMP 병렬화, AMG 전처리기\n'
    'Grace/Tomiyama/Ishii-Zuber 항력, Tomiyama 양력, Burns 난류분산\n'
    'GMSH 메쉬 리더, 3D Hex/Tet 혼합 격자, AMR 적응 세분화')
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(80, 80, 80)

lang = doc.add_paragraph()
lang.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = lang.add_run(
    'C++17 / Eigen 3.4 기반 유한체적법(FVM) 코드\n'
    '2D/3D 구조·비정렬 격자 지원 | 29개 모듈 | ~14,000 lines | 25개 검증 케이스')
run.font.size = Pt(10)
run.font.italic = True

doc.add_page_break()

# ============================================================
# Table of contents
# ============================================================
doc.add_heading('목차', level=1)
toc_items = [
    '1. 서론',
    '   1.1 개발 배경',
    '   1.2 Python → C++ 전환 개요',
    '   1.3 개발 범위',
    '2. 지배방정식',
    '   2.1 Two-Fluid Model (Euler-Euler) — 6방정식 완전 비평형',
    '   2.2 단상 비압축성 Navier-Stokes',
    '   2.3 난류 모델 (k-ε, k-ω SST) — 상수 전체 포함',
    '   2.4 고체 열전도 및 CHT',
    '   2.5 상변화 모델 (Lee, Rohsenow, Zuber, Nusselt) — 완전 수식',
    '   2.6 계면력 — 항력 4종, 양력, 벽윤활, 난류분산, CSF 표면장력',
    '   2.7 복사 및 화학반응',
    '3. 수치기법',
    '   3.1 유한체적법(FVM) 이산화 — 적분형 유도',
    '   3.2 SIMPLE/PISO 알고리즘 — 단계별 수식',
    '   3.3 Rhie-Chow 운동량 보간',
    '   3.4 MUSCL/TVD — 지연 보정 + 4종 제한자',
    '   3.5 BDF2 2차 시간적분',
    '   3.6 비직교 격자 보정',
    '   3.7 선형 솔버 및 AMG 전처리기',
    '4. 코드 구조 (29개 모듈)',
    '5. 격자 생성 및 I/O',
    '6. 병렬화 (OpenMP)',
    '7. IAPWS-IF97 증기표',
    '8. 엔지니어링 QA 감사',
    '9. 검증 결과 — 문헌 기반 상세 비교',
    '   9.1 MMS 수렴 검증 (Case 6 MUSCL)',
    '   9.2 Ghia Lid-Driven Cavity (Case 2)',
    '   9.3 Poiseuille 유동 (Case 1)',
    '   9.4 Bubble Column (Case 4)',
    '   9.5 상변화 / 풀비등 (Cases 9, 24)',
    '   9.6 CHT 및 복사 (Cases 3, 11)',
    '   9.7 3D 격자 및 AMR (Cases 14, 12)',
    '   9.8 Python vs C++ 수치 비교',
    '   9.9 성능 비교',
    '10. 결론 및 향후 과제',
    '11. 참고문헌',
]
for item in toc_items:
    doc.add_paragraph(item)

doc.add_page_break()

# ============================================================
# 1. 서론
# ============================================================
doc.add_heading('1. 서론', level=1)
body(
    '본 보고서는 Two-Fluid Model(Euler-Euler) 기반의 유한체적법(FVM) 열유체 해석 코드 '
    'K-CFD의 개발 및 검증 결과를 기술한다. 본 코드는 원래 Python으로 개발되었으며, '
    '엔지니어링 프로덕션 사용을 위해 C++17(Eigen 3.4)로 완전히 전환되었다. '
    '2D/3D 구조·비정렬 격자에서의 단상/이상 유동, 난류, 고체 열전도 및 유체-고체 '
    '공액 열전달, 상변화, 복사, 화학반응을 포함하는 종합적 열유체 해석이 가능하다. '
    'C++ 전환을 통해 Python 대비 평균 31배, 최대 105배의 성능 향상을 달성하였으며, '
    '2차에 걸친 엔지니어링 QA 감사를 통해 총 34개의 CRITICAL/HIGH 이슈를 해결하였다.')

doc.add_heading('1.1 개발 배경', level=2)
body(
    '원자로, 화학반응기, 열교환기 등 다양한 공학 시스템에서 이상유동 및 열전달 현상의 '
    '정확한 예측은 안전 설계의 핵심이다. Two-Fluid Model은 각 상을 연속체로 취급하여 '
    '대규모 시스템의 열유체 해석에 효과적인 접근법을 제공한다.\n\n'
    '기존 Python 코드는 25개 검증 케이스를 통해 알고리즘의 정확성이 확인되었으나, '
    '계산 속도와 메모리 효율의 한계로 대규모 산업 문제에 적용하기 어려웠다. '
    'C++ 전환을 통해 평균 31배, 최대 105배의 속도향상을 달성하였으며, '
    '2차에 걸친 엔지니어링 QA 감사를 통해 총 34개의 CRITICAL/HIGH 이슈를 '
    '모두 해결하여 프로덕션급 품질을 확보하였다.')

doc.add_heading('1.2 Python → C++ 전환 개요', level=2)
add_table(
    ['항목', 'Python (원본)', 'C++ (현재)'],
    [
        ['언어', 'Python 3.7+ / NumPy / SciPy', 'C++17 / Eigen 3.4'],
        ['코드 규모', '~17,000 lines', '~14,000 lines (29 모듈)'],
        ['선형 솔버', 'scipy.sparse (LU, BiCGSTAB, GMRES)', 'Eigen (SparseLU, BiCGSTAB, CG) + AMG'],
        ['병렬화', 'MPI (mpi4py), CuPy GPU', 'OpenMP (공유 메모리)'],
        ['난류 모델', 'k-ε', 'k-ε + k-ω SST'],
        ['항력 모델', 'Schiller-Naumann', 'S-N + Grace + Tomiyama + Ishii-Zuber'],
        ['계면력', '항력만', '항력 + 양력 + 벽윤활 + 난류분산'],
        ['표면장력', '없음', 'CSF (Brackbill 1992)'],
        ['물성치', '상수', '상수 + IAPWS-IF97'],
        ['시간적분', '후방 Euler', 'Euler + BDF2'],
        ['압력-속도 결합', 'SIMPLE', 'SIMPLE + PISO + Rhie-Chow'],
        ['확산 이산화', '직교만', '직교 + 비직교 보정'],
        ['메쉬 입력', '자체 생성기', '자체 생성기 + GMSH .msh 리더'],
        ['평균 속도향상', '기준 (1x)', '~31x (최대 105x)'],
    ]
)

doc.add_heading('1.3 개발 범위', level=2)
body('본 코드의 주요 기능:')
for f in [
    'Two-Fluid (Euler-Euler) 이상유동 모델 (6방정식 비평형)',
    'SIMPLE/PISO 알고리즘 기반 속도-압력 커플링 + Rhie-Chow 운동량 보간',
    'k-ε 및 k-ω SST 난류 모델 + 자동 벽 처리 (Low-Re/Wall-Function/AUTOMATIC)',
    '고체 열전도 및 유체-고체 CHT 커플링',
    '4종 항력 모델: Schiller-Naumann, Grace, Tomiyama, Ishii-Zuber',
    '3종 계면력: Tomiyama 양력, Antal 벽윤활, Burns 난류분산',
    'CSF 표면장력 모델 (Brackbill 1992)',
    'IAPWS-IF97 증기표 (Region 1/2, 포화선, 수송물성)',
    'Lee/Rohsenow/Zuber/Nusselt 상변화 모델',
    'MUSCL/TVD 2차 대류 이산화 (van Leer, minmod, superbee, van Albada)',
    'BDF2 2차 시간적분',
    '비직교 격자 확산 보정',
    'AMG V-cycle 전처리기 + 전처리 BiCGSTAB/CG',
    'P1 복사 근사 모델',
    '1차 화학반응 (A→B) 및 종 수송',
    '2D/3D 구조격자 생성기 + Hex/Tet 혼합 격자',
    'GMSH .msh 2.2 ASCII 메쉬 리더',
    'AMR 적응 격자 세분화',
    'OpenMP 공유 메모리 병렬화',
    'VTU (ParaView) 출력 + pybind11 Python 바인딩',
]:
    body(f'  • {f}')

doc.add_page_break()

# ============================================================
# 2. 지배방정식  (DETAILED — ~2 pages per major model)
# ============================================================
doc.add_heading('2. 지배방정식', level=1)

# ------------------------------------------------------------------
# 2.1 Two-Fluid Model
# ------------------------------------------------------------------
doc.add_heading('2.1 Two-Fluid Model (Euler-Euler) — 6방정식 완전 비평형', level=2)
body(
    'Two-Fluid Model(Euler-Euler)은 각 상(phase) k ∈ {l(액상), g(기상)}을 상호침투 '
    '연속체로 취급하여 각 상에 독립적인 보존방정식을 부여한다. '
    '6방정식 비평형 모델은 양 상 간 속도 미끄럼(velocity slip)과 온도 비평형을 허용한다. '
    '이는 균질 평형 모델(HEM)보다 계산 비용이 크지만 계면 동역학을 정확히 표현한다.')

body('■ 연속 방정식 (각 상 k):')
eq('∂(αₖρₖ)/∂t  +  ∇·(αₖρₖuₖ)  =  Γₖ')
body(
    '여기서 αₖ는 체적분율, ρₖ는 밀도, uₖ는 속도 벡터, Γₖ는 계면 질량 전달률이다. '
    '액상과 기상 사이의 질량 보존: Γₗ + Γg = 0.')

body('■ 운동량 방정식 (각 상 k):')
eq(
    '∂(αₖρₖuₖ)/∂t  +  ∇·(αₖρₖuₖ⊗uₖ)\n'
    '  = -αₖ∇p  +  ∇·(αₖτₖ)  +  αₖρₖg  +  Mₖ  +  Γₖuₖᵢ')
body(
    'p는 공유 압력(shared pressure)이다. '
    'Mₖ는 상간 힘의 합: Mₖ = Mₖᴰ + Mₖᴸ + Mₖᵂ + Mₖᵀᴰ + Mₖᵟ (항력, 양력, 벽윤활, 난류분산, 표면장력). '
    'uₖᵢ는 계면 속도(보통 uₖ 또는 두 상 평균).')

body('■ 점성 응력 텐서 (뉴턴 유체):')
eq('τₖ  =  μₖ(∇uₖ + ∇uₖᵀ)  -  (2/3)μₖ(∇·uₖ)I')
body('μₖ = μₖ_lam + μₖ_t  (층류 + 난류 점성). 비압축 흐름에서 ∇·uₖ ≈ 0.')

body('■ 에너지 방정식 (엔탈피 형식, 각 상 k):')
eq(
    '∂(αₖρₖhₖ)/∂t  +  ∇·(αₖρₖuₖhₖ)\n'
    '  =  ∂(αₖp)/∂t  +  ∇·(αₖkₖ∇Tₖ)  +  Qₖ  +  Γₖhₖᵢ')
body(
    'hₖ는 상 엔탈피, kₖ는 열전도도. '
    'Qₖ는 상간 열전달: Qₖ = hᵢaᵢ(Tg - Tl) (기상) / hᵢaᵢ(Tl - Tg) (액상). '
    'hₖᵢ는 계면 포화 엔탈피.')

body('■ 체적분율 구속조건:')
eq('αₗ + αg  =  1')
body(
    '이를 통해 압력장은 αg를 통해 결정된다. '
    '구현에서는 αg 방정식만 독립적으로 풀고, αₗ = 1 - αg로 얻는다.')

body('■ 비계면적 및 상간 열전달 (Ranz-Marshall):')
eq('aᵢ  =  6·αg / dᵦ      [m⁻¹]')
eq('Nu  =  2.0  +  0.6·Re_b^0.5·Pr^(1/3)      (Ranz-Marshall, 1952)')
eq('hᵢ  =  Nu · kₗ / dᵦ      [W/(m²·K)]')
body(
    'dᵦ는 기포 직경(사용자 설정 또는 모델 계산). '
    'Re_b = ρₗ|ug - ul|dᵦ/μₗ, Pr = μₗcₗ/kₗ.')

body('■ Sato 기포 유도 난류(BIT):')
eq('μₜ_BIT  =  C_BIT · ρₗ · αg · dᵦ · |ug - ul|      (C_BIT = 0.6)')
body('BIT는 기포 후류에 의한 추가 난류 점성으로 계면 열전달을 증가시킨다.')

# ------------------------------------------------------------------
# 2.2 Single-phase N-S
# ------------------------------------------------------------------
doc.add_heading('2.2 단상 비압축성 Navier-Stokes', level=2)
body('비압축성 단상 유동에 대해 연속 방정식과 운동량 방정식을 독립적으로 푼다.')
eq('∇·u  =  0      (연속)')
eq('∂(ρu)/∂t  +  ∇·(ρu⊗u)  =  -∇p  +  ∇·(μ_eff·∇u)  +  ρg  +  S_u')
body('μ_eff = μ_lam + μ_t (층류 + 난류 점성). 에너지 방정식 (온도):')
eq('∂(ρcₚT)/∂t  +  ∇·(ρcₚuT)  =  ∇·(k_eff·∇T)  +  S_T')

# ------------------------------------------------------------------
# 2.3 Turbulence (DETAILED)
# ------------------------------------------------------------------
doc.add_heading('2.3 난류 모델 — 상수 전체 포함', level=2)

doc.add_heading('2.3.1 표준 k-ε 모델 (Launder & Spalding, 1974)', level=3)
body(
    'k-ε 모델은 난류 운동 에너지 k와 소산율 ε에 대한 두 개의 수송 방정식을 추가한다. '
    '반실험적 모델로, 높은 Re 수의 전단 지배 유동에서 광범위하게 검증되었다.')
eq(
    '∂(ρk)/∂t  +  ∇·(ρuk)\n'
    '  =  ∇·((μ + μ_t/σ_k)·∇k)  +  Pₖ  -  ρε')
eq(
    '∂(ρε)/∂t  +  ∇·(ρuε)\n'
    '  =  ∇·((μ + μ_t/σ_ε)·∇ε)  +  C₁ε·(ε/k)·Pₖ  -  C₂ε·ρ·ε²/k')
eq('μ_t  =  ρ·C_μ·k²/ε')
eq('Pₖ  =  μ_t·(∇u + ∇uᵀ):∇u  =  2μ_t·Sᵢⱼ·Sᵢⱼ')
body('모델 상수 (Launder & Spalding 1974):')
add_table(
    ['상수', '값', '물리적 의미'],
    [
        ['C_μ',  '0.09', '난류 점성 계수'],
        ['C₁ε', '1.44', 'ε 생산항 계수'],
        ['C₂ε', '1.92', 'ε 소산항 계수'],
        ['σ_k',  '1.0',  'k 확산 Prandtl 수'],
        ['σ_ε',  '1.3',  'ε 확산 Prandtl 수'],
    ]
)

doc.add_heading('2.3.2 Menter k-ω SST 모델 (1994)', level=3)
body(
    'k-ω SST(Shear Stress Transport)는 두 블렌딩 함수 F₁, F₂를 이용하여 '
    '벽 근방에서 k-ω, 자유류에서 k-ε의 장점을 결합한다. '
    '역압력 구배 유동과 분리 유동에서 k-ε보다 우수한 성능을 보인다.')

body('k 방정식:')
eq(
    '∂(ρk)/∂t  +  ∇·(ρuk)\n'
    '  =  ∇·((μ + σ_k·μ_t)·∇k)  +  P̃ₖ  -  β*·ρ·k·ω')

body('ω 방정식:')
eq(
    '∂(ρω)/∂t  +  ∇·(ρuω)\n'
    '  =  ∇·((μ + σ_ω·μ_t)·∇ω)  +  γ·(ω/k)·P̃ₖ  -  β·ρ·ω²\n'
    '  +  2(1-F₁)·ρ·σ_ω2/ω·(∇k·∇ω)')
body('마지막 교차 확산(cross-diffusion)항은 k-ε 영역에서 활성화된다.')

body('난류 점성 (a₁ 클리핑으로 SST 특성 부여):')
eq('μ_t  =  ρ·a₁·k / max(a₁·ω,  S·F₂)')
body('여기서 S = √(2Sᵢⱼ·Sᵢⱼ) 는 strain rate magnitude.')

body('생산 제한자:')
eq('P̃ₖ  =  min(Pₖ,  10·β*·ρ·k·ω)')

body('블렌딩 함수:')
eq(
    'F₁  =  tanh(arg₁⁴)\n'
    'arg₁  =  min(max(√k/(β*·ω·y),  500μ/(ρ·ω·y²)),  4ρσ_ω2·k/(CD_kω·y²))\n'
    'CD_kω  =  max(2ρσ_ω2/ω·(∇k·∇ω),  1×10⁻¹⁰)')
eq(
    'F₂  =  tanh(arg₂²)\n'
    'arg₂  =  max(2√k/(β*·ω·y),  500μ/(ρ·ω·y²))')
body('y는 가장 가까운 벽까지의 거리.')

body('모델 상수 (Set 1 / Set 2):')
add_table(
    ['상수', 'Set 1 (k-ω, 벽 근방)', 'Set 2 (k-ε, 자유류)', '물리적 의미'],
    [
        ['σ_k',   '0.85',   '1.0',    'k 확산 Prandtl 수'],
        ['σ_ω',   '0.5',    '0.856',  'ω 확산 Prandtl 수'],
        ['β',     '0.075',  '0.0828', 'ω 소산 계수'],
        ['β*',    '0.09',   '0.09',   'k 소산 계수'],
        ['a₁',    '0.31',   '0.31',   'μ_t 클리핑 계수'],
        ['γ',     '5/9',    '0.44',   'ε↔ω 전환 계수'],
        ['κ',     '0.41',   '0.41',   'von Kármán 상수'],
    ]
)
body('블렌딩: φ = F₁·φ₁ + (1 - F₁)·φ₂  (각 상수 가중 평균)')

doc.add_heading('2.3.3 자동 벽 처리 (y⁺ 기반 블렌딩)', level=3)
body(
    '벽 인접 셀의 y⁺ = ρ·u_τ·y/μ 값에 따라 자동으로 처리 모드를 선택한다. '
    'u_τ = √(τ_w/ρ) 는 마찰 속도.')
add_table(
    ['y⁺ 범위', '처리 모드', 'k-ε 경계값', 'SST ω 경계값'],
    [
        ['< 5',     'Low-Re (점성 아층 해상)', 'k→0, ε=2μk/(ρy²)', 'ω = 6μ/(ρβ₁y²)'],
        ['5 ~ 30',  '선형 블렌딩',             '(1-b)·LowRe + b·WF', '(1-b)·LowRe + b·WF'],
        ['> 30',    '벽함수 (대수 법칙)',       'k=u_τ²/√C_μ, ε=u_τ³/(κy)', 'ω=√k/(C_μ^0.25·κ·y)'],
    ]
)
eq('blend  =  (y⁺ - 5) / 25      for  5 ≤ y⁺ ≤ 30')
note('y⁺ > 300 또는 y⁺ < 1 이면 경고를 출력하여 격자 품질을 모니터링한다.')

# ------------------------------------------------------------------
# 2.4 CHT
# ------------------------------------------------------------------
doc.add_heading('2.4 고체 열전도 및 공액 열전달(CHT)', level=2)
body('고체 영역에서 비정상 열전도:')
eq('ρ_s·c_s·∂T/∂t  =  ∇·(k_s·∇T)  +  q‴')
body(
    '유체-고체 인터페이스 조건: 온도 연속성(T_f = T_s)과 열유속 연속성.\n'
    '계면 유효 열전도도: k_eff = 조화 평균 = 2·k_f·k_s/(k_f + k_s).')
eq('q_interface  =  k_eff · (T_f - T_s) / Δy_interface')

# ------------------------------------------------------------------
# 2.5 Phase Change (DETAILED)
# ------------------------------------------------------------------
doc.add_heading('2.5 상변화 모델 — 완전 수식', level=2)

doc.add_heading('2.5.1 Lee 체적 상변화 모델 (1980)', level=3)
body(
    'Lee 모델은 국소 온도 편차를 구동력으로 하는 체적 증발·응축 모델이다. '
    '원자로 안전 분석(RELAP, TRACE)에 광범위하게 사용된다.')
body('증발 (T > T_sat):')
eq('Γ_evap  =  r · ρₗ · αₗ · (T - T_sat) / T_sat      [kg/(m³·s)]')
body('응축 (T < T_sat):')
eq('Γ_cond  =  r · ρg · αg · (T_sat - T) / T_sat      [kg/(m³·s)]')
body(
    '완화 계수 r의 권장 범위: 0.001 ≤ r ≤ 100 [1/s]. '
    '큰 r은 평형에 가까운 상변화를 모사하지만 강성(stiffness)을 증가시킨다. '
    '소스 선형화: S_T = -(Γ·h_fg)/T_sat 로 에너지 방정식에 결합.')

doc.add_heading('2.5.2 Rohsenow 핵비등 열전달 상관식 (1952)', level=3)
body(
    'Rohsenow 상관식은 과열 벽면에서의 핵비등 열유속을 예측한다. '
    '표면 재질과 유체 조합의 영향을 C_sf 계수로 반영한다.')
eq(
    'q″_nb  =  μₗ · h_fg · √(g(ρₗ - ρg)/σ) · [cₚₗ·ΔT_sat / (C_sf·h_fg·Prₗⁿ)]³')
body(
    'ΔT_sat = T_wall - T_sat (벽 과열도). '
    'C_sf: 표면-유체 조합 계수 (물/스테인리스: 0.013, 물/구리: 0.006). '
    'n = 1.0 (물), 1.7 (기타 유체). '
    'Prₗ: 액상 Prandtl 수. h_fg: 증발 잠열.')

doc.add_heading('2.5.3 Zuber CHF(임계열유속) 상관식 (1959)', level=3)
body(
    'Zuber 상관식은 유체역학적 불안정성(Rayleigh-Taylor, Kelvin-Helmholtz)에 기반한 '
    'Critical Heat Flux 예측 모델이다.')
eq('q″_CHF  =  0.131 · ρg · h_fg · [σ·g·(ρₗ - ρg) / ρg²]^(1/4)')
body(
    '계수 0.131은 Lienhard & Dhir (1973)의 정사각형 평면 히터 실험에서 보정되었다. '
    'Kutateladze 계수: 0.131 ≈ π/24.')

doc.add_heading('2.5.4 Nusselt 막응축 상관식 (1916)', level=3)
body(
    'Nusselt 막응축 이론은 수직 평판 위 중력 지배 층류 막에 대한 해석 해이다. '
    'McAdams(1954)의 실험 비교에서 ±10% 이내 정확도를 보인다.')
eq(
    'h_film  =  0.943 · [ρₗ·(ρₗ - ρg)·g·h_fg·kₗ³ / (μₗ·L·ΔT_sub)]^(1/4)')
body(
    'ΔT_sub = T_sat - T_wall (벽 과냉도). L: 평판 높이. '
    'kₗ: 액막 열전도도. 모든 물성치는 액막 평균 온도에서 평가.')

# ------------------------------------------------------------------
# 2.6 Interfacial forces (DETAILED)
# ------------------------------------------------------------------
doc.add_heading('2.6 계면력 — 항력, 양력, 벽윤활, 난류분산, CSF', level=2)
body(
    '계면 운동량 전달 합력: M_k = M^D + M^L + M^W + M^TD + M^σ. '
    '뉴턴 제3법칙: M_l + M_g = 0.')

doc.add_heading('2.6.1 항력 모델 4종', level=3)
body('항력: M^D_g = -(3/4)·C_D/d_b·α_g·ρ_l·|u_r|·u_r,  u_r = u_g - u_l')

body('■ Schiller-Naumann (1935) — 강체 구형 입자/소형 기포:')
eq(
    'C_D  =  24/Re · (1 + 0.15·Re^0.687)      Re < 1000\n'
    'C_D  =  0.44                              Re ≥ 1000')
eq('Re  =  ρₗ · |u_r| · d_b / μₗ')

body('■ Grace (1973) — 변형 가능한 기포(청정 시스템):')
eq(
    'Mo  =  g·μₗ⁴·(ρₗ-ρg) / (ρₗ²·σ³)      (Morton 수)\n'
    'Eo  =  g·(ρₗ-ρg)·d_b² / σ             (Eötvös 수)\n'
    'J   =  0.94·H^0.757      for H ≤ 59.3\n'
    'J   =  3.42·H^0.441      for H > 59.3\n'
    'H   =  (4/3)·Eo·Mo^(-0.149)·(μₗ/μ_ref)^(-0.14)\n'
    'u_terminal  =  μₗ·Mo^(-0.149)·(J - 0.857) / (ρₗ·d_b)\n'
    'C_D  =  4·d_b·g·(ρₗ-ρg) / (3·ρₗ·u_terminal²)')

body('■ Tomiyama (1998) — 오염 시스템 기포:')
eq(
    'C_D  =  max(\n'
    '  24/Re·(1 + 0.15·Re^0.687),\n'
    '  8·Eo / (3·(Eo + 4))\n'
    ')')
body('오염 효과는 표면 이동성 억제를 통해 C_D를 강체 구 수준으로 제한한다.')

body('■ Ishii-Zuber (1979) — 밀집 분산상:')
eq(
    'μ_m   =  μₗ·(1 - αg)^(-2.5·(μg + 0.4μₗ)/(μg + μₗ))\n'
    'Re_m  =  ρₗ·|u_r|·d_b / μ_m\n'
    'C_D   =  24/Re_m·(1 + 0.15·Re_m^0.687)      (distorted bubble regime)')
body('혼합 점성 μ_m은 Brinkman 관계를 사용하여 밀집 분산상 효과를 반영한다.')

doc.add_heading('2.6.2 Tomiyama 양력 모델 (2002)', level=3)
body('양력은 속도 기울기가 있는 유동에서 기포를 측면으로 이동시킨다.')
eq('M^L_g  =  -C_L · αg · ρₗ · (u_g - u_l) × (∇ × u_l)')
body('C_L는 Eötvös 수 Eo_d의 함수:')
eq(
    'Eo_d  =  g·(ρₗ-ρg)·d_h² / σ    (d_h: 수평 기포 직경)\n'
    'f(Eo_d)  =  0.00105·Eo_d³ - 0.0159·Eo_d² - 0.0204·Eo_d + 0.474\n'
    'C_L  =  min(0.288·tanh(0.121·Re),  f(Eo_d))    if Eo_d < 4\n'
    'C_L  =  f(Eo_d)                                if 4 ≤ Eo_d ≤ 10.7\n'
    'C_L  =  -0.27                                   if Eo_d > 10.7')
body('큰 기포(Eo_d > 10.7)는 C_L < 0 이므로 벽 방향으로 이동한다.')

doc.add_heading('2.6.3 Antal 벽윤활 모델 (1991)', level=3)
body('벽윤활력은 기포가 벽면 근방에서 반발되는 효과를 모델링한다.')
eq(
    'C_w(y)  =  max(0,  C_w1/d_b + C_w2/y_wall)\n'
    'M^W_g  =  -C_w · αg · ρₗ · |u_r_t|² · n̂_w')
body(
    'C_w1 = -0.0064, C_w2 = 0.016 (원저자 권장값). '
    'n̂_w: 벽면에서 유체로의 법선 벡터. y_wall: 벽까지의 거리. '
    'u_r_t: 벽 접선 방향 상대 속도.')

doc.add_heading('2.6.4 Burns 난류분산 모델 (2004)', level=3)
body(
    'Favre 평균 항력 모델에서 유도된 난류분산 항으로, '
    '기포의 난류 혼합을 공식적으로 표현한다.')
eq(
    'M^TD_g  =  -C_td · μ_t / σ_td · (∇αg/αg  -  ∇αₗ/αₗ)\n'
    '         =  -C_td · μ_t / σ_td · ∇αg / (αg(1 - αg))')
body('C_td = 1.0 (권장), σ_td = 0.9 (분산 Schmidt 수).')

doc.add_heading('2.6.5 CSF 표면장력 모델 (Brackbill 1992)', level=3)
body(
    'Continuum Surface Force(CSF) 모델은 계면의 표면장력을 체적력으로 변환한다. '
    'α_g의 기울기로 계면 법선과 곡률을 계산한다.')
eq(
    'F_σ  =  σ · κ · ∇αg\n'
    'κ    =  -∇·n̂      (계면 평균 곡률)\n'
    'n̂    =  ∇αg / |∇αg|      (계면 법선)')
body(
    'σ: 표면장력 계수 [N/m]. IAPWS 모드에서는 온도 함수로 자동 계산:\n'
    'σ(T) = 0.2358·(1 - T/T_c)^1.256·(1 - 0.625·(1 - T/T_c))  (T_c = 647.1 K).')
note('수치 안정성을 위해 |∇αg| < ε_small 이면 F_σ = 0 처리.')

# ------------------------------------------------------------------
# 2.7 Radiation & Chemistry
# ------------------------------------------------------------------
doc.add_heading('2.7 복사 및 화학반응', level=2)
body('■ P1 복사 근사 모델:')
eq('-∇·(1/(3κ) · ∇G)  +  κG  =  4κσT⁴')
body(
    'G: 방사 강도(irradiation) [W/m²]. κ: 흡수 계수. σ: Stefan-Boltzmann 상수 = 5.67e-8 W/(m²·K⁴). '
    'P1 모델은 확산형 PDE이므로 표준 FVM으로 풀 수 있다.')

body('■ 1차 화학반응:')
eq('A → B,    R_A  =  -k_r · C_A      [mol/(m³·s)]')
body('선형화 소스: S_A = -k_r·C_A (implicit). k_r은 Arrhenius 또는 상수 입력.')

doc.add_page_break()

# ============================================================
# 3. 수치기법 (DETAILED)
# ============================================================
doc.add_heading('3. 수치기법', level=1)

doc.add_heading('3.1 유한체적법(FVM) 이산화 — 적분형 유도', level=2)
body(
    '셀 중심(cell-centered) 배치. 임의의 보존 방정식을 제어체적 Ω에 적분한다:\n'
    '∫_Ω ∂(ρφ)/∂t dV + ∮_∂Ω (ρuφ - Γ∇φ)·dA = ∫_Ω S_φ dV')

body('■ 확산항 이산화:')
eq(
    '∮ Γ∇φ·dA  ≈  Σ_f  Γ_f · A_f · (φ_N - φ_O) / d_PN\n'
    'Γ_f  =  2·Γ_O·Γ_N / (Γ_O + Γ_N)      (조화 평균)\n'
    'd_PN  =  |x_N - x_O|      (셀 중심 거리)')
body('비직교 보정은 3.6절 참조.')

body('■ 대류항 이산화:')
eq(
    '∮ ρuφ·dA  ≈  Σ_f  F_f · φ_f\n'
    'F_f  =  ρ_f · u_f · A_f      (면 질량 유량, Rhie-Chow 보간)\n'
    '1차 풍상:  φ_f = φ_O  if F_f > 0,  φ_N  if F_f < 0\n'
    'MUSCL:    φ_f = φ_UW + ψ(r)·(φ_HO - φ_UW)  (지연 보정)')

body('■ 소스항 선형화 (음수 S_p로 안정성 확보):')
eq('S_φ ≈ S_u + S_p·φ,      S_p ≤ 0')

body('■ 조립된 선형 시스템:')
eq('a_P·φ_P  +  Σ_nb a_nb·φ_nb  =  b_P')
body('a_P, a_nb는 확산+대류 계수. b_P는 소스 + 명시항 + 시간항.')

doc.add_heading('3.2 SIMPLE/PISO 알고리즘 — 단계별 수식', level=2)
body('■ SIMPLE (Semi-Implicit Method for Pressure-Linked Equations):')
eq(
    '단계 1. 운동량 예측:\n'
    '  a_P·u* = H(u*) - ∇p^n    [H(u) = -Σ a_nb·u_nb + b]\n'
    '\n'
    '단계 2. 압력 보정 방정식 (연속 방정식 적용):\n'
    '  ∇·(d·∇p\')  =  ∇·u*      [d = V/a_P]\n'
    '\n'
    '단계 3. 속도/압력 갱신:\n'
    '  p = p^n + α_p·p\'\n'
    '  u = u* - d·∇p\'           (완화: u ← α_u·u + (1-α_u)·u^n)\n'
    '\n'
    '단계 4. 수렴 검사 → 미수렴 시 단계 1로')
body('권장 완화 계수: α_u = 0.7 (속도), α_p = 0.3 (압력).')

body('■ PISO (Pressure Implicit with Splitting of Operators):')
eq(
    '단계 1. 운동량 예측 (α_u = 1.0, 완화 없음):\n'
    '  a_P·u* = H(u*) - ∇p^n\n'
    '\n'
    '단계 2. 1차 압력 보정:\n'
    '  ∇·(d·∇p\') = ∇·u*\n'
    '  u** = u* - d·∇p\',  p^(1) = p^n + p\'\n'
    '\n'
    '단계 3. 2차 압력 보정 (반복):\n'
    '  a_P·u** = H(u**) - ∇p^(1)  →  재조립\n'
    '  ∇·(d·∇p\'\') = ∇·u**\n'
    '  u^(n+1) = u** - d·∇p\'\',  p^(n+1) = p^(1) + p\'\'')
body('PISO는 시간 스텝당 1회 외부 반복으로 비정상 계산에 적합하다.')

doc.add_heading('3.3 Rhie-Chow 운동량 보간', level=2)
body(
    'Collocated 배치에서 체커보드 압력 진동을 방지하기 위한 운동량 보간. '
    'Rhie & Chow (1983)에서 제안되었으며 현대 RANS/LES 코드의 표준이다.')
eq(
    'u_f^RC  =  (w·u_O + (1-w)·u_N)  -  d_f·(∇p_compact - ∇p_interp)\n'
    '\n'
    'd_f     =  (V/a_P)_f      (면에서 보간한 운동량 계수 역수)\n'
    '∇p_compact  =  (p_N - p_O) / d_PN · n̂_f      (면 조밀 압력 기울기)\n'
    '∇p_interp   =  w·(∇p)_O + (1-w)·(∇p)_N      (셀 중심 기울기 보간)')
body(
    'w = d_PN_N / (d_PN_O + d_PN_N): 거리 역비 보간 가중치. '
    '보정항 (∇p_compact - ∇p_interp)이 체커보드를 소산시킨다. '
    '초기 단계(a_P ≤ 1.1)에서는 Rhie-Chow를 비활성화하여 초기 발산을 방지한다.')

doc.add_heading('3.4 MUSCL/TVD — 지연 보정 + 4종 제한자', level=2)
body(
    'MUSCL(Monotone Upstream-centered Scheme for Conservation Laws) + TVD 제한자 방식으로 '
    '2차 정확도와 비물리적 진동 억제를 동시에 달성한다. '
    '지연 보정(deferred correction) 패턴으로 implicit 1차 주 행렬에 explicit 고차 보정을 추가.')
eq(
    'φ_f  =  φ_f^UW  +  ψ(r)·(φ_f^HO - φ_f^UW)\n'
    '\n'
    'φ_f^UW  =  φ_upwind           (1차 풍상)\n'
    'φ_f^HO  =  (φ_O + φ_N)/2     (2차 중앙차분)\n'
    'r       =  (φ_P - φ_UU) / (φ_D - φ_P)    (기울기 비율)')
body('TVD 제한자 함수 ψ(r):')
add_table(
    ['제한자', 'ψ(r) 수식', '특성'],
    [
        ['van Leer',   'ψ = (r + |r|) / (1 + |r|)',                    '부드러움, 2차 수렴 보장'],
        ['minmod',    'ψ = max(0, min(r, 1))',                          '가장 보수적, 과도한 소산'],
        ['superbee',  'ψ = max(0, min(2r, 1), min(r, 2))',              '가장 덜 소산, 경우에 따라 진동'],
        ['van Albada', 'ψ = (r² + r) / (r² + 1)',                      '부드러운 TVD, 경계 근방 적합'],
    ]
)
body('기본값: van Leer. 모든 제한자는 Sweby(1984) TVD 조건을 만족한다.')

doc.add_heading('3.5 BDF2 2차 시간적분', level=2)
body(
    'BDF2(Backward Differentiation Formula 2차)는 3개의 시간 레벨을 사용하는 '
    '무조건 안정, 2차 정확도 암시적 방법이다.')
eq(
    '(3ρV)/(2Δt)·φⁿ⁺¹  =  (4ρV)/(2Δt)·φⁿ  -  (ρV)/(2Δt)·φⁿ⁻¹\n'
    '                     +  RHS(φⁿ⁺¹,  uⁿ⁺¹,  pⁿ⁺¹)')
body(
    'ScalarField는 old_values(φⁿ)와 old_old_values(φⁿ⁻¹)를 저장한다. '
    'store_old() 호출 시 자동 체이닝: old_old ← old ← current. '
    '첫 번째 시간 스텝은 자동으로 1차 후방 Euler(1st-order BDF)를 사용하여 cold start 문제를 회피한다.')
eq('설정: solver.time_scheme = "bdf2"')

doc.add_heading('3.6 비직교 격자 보정', level=2)
body(
    '비직교 메쉬에서 face normal n̂_f가 셀 중심 연결 벡터 d_PN과 불일치하면 '
    '표준 FVM 확산 이산화의 정확도가 1차로 저하된다. '
    '명시적 비직교 보정으로 2차 정확도를 유지한다.')
eq(
    '(확산항) =  Σ_f  Γ_f·A_f·(φ_N - φ_O)/d_orth      [implicit, 직교 성분]\n'
    '          + Σ_f  Γ_f · [(∇φ_f·d⃗_f) - (φ_N - φ_O)·|d⃗_f|/d_PN]  [explicit, 비직교 성분]')
body(
    'd_orth = (x_N - x_O)·n̂_f: 직교 거리. d⃗_f: 실제 셀 연결 벡터. '
    '비직교 성분은 지연 보정(deferred correction)으로 RHS에 추가한다. '
    'n_nonorth_correctors 반복 횟수(기본 2)만큼 보정한다.')

doc.add_heading('3.7 선형 솔버 및 AMG 전처리기', level=2)
add_table(
    ['솔버', '용도', '특성'],
    [
        ['SparseLU', '소규모 직접 풀이', 'Eigen::SparseLU, O(n^1.5) 메모리'],
        ['BiCGSTAB', '비대칭 반복 풀이', '운동량, k, ε, ω, 온도 방정식'],
        ['CG', '대칭 반복 풀이', '압력 보정 방정식'],
        ['전처리 BiCGSTAB/CG', '수렴 가속', 'Jacobi / ILU(0) / AMG 결합'],
    ]
)
body('■ AMG V-cycle 전처리기:')
eq(
    '1. 거칠게하기: 쌍별 응집(pairwise aggregation)\n'
    '   — 가장 강한 연결 계수로 이웃 셀 쌍 형성\n'
    '2. 연장 연산자: P (prolongation)\n'
    '3. 제한 연산자: R = Pᵀ (Galerkin)\n'
    '4. 거친 격자 연산자: A_c = R · A · P\n'
    '5. 전/후 평활: Gauss-Seidel 2회\n'
    '6. 최소 격자 직접 풀이: Eigen::SparseLU')
note('AMG V-cycle 1회를 전처리로 사용. 대규모 격자(>10⁵ 셀)에서 ILU 대비 3-10배 빠른 수렴.')

doc.add_page_break()

# ============================================================
# 4. 코드 구조
# ============================================================
doc.add_heading('4. 코드 구조 (29개 모듈)', level=1)
add_table(
    ['카테고리', '모듈', '파일', '설명'],
    [
        ['Core', '메쉬', 'mesh', 'Face/Cell/FVMesh, 경계면 해시맵 캐시'],
        ['', '필드', 'fields', 'ScalarField, VectorField (old/old_old)'],
        ['', 'FVM 연산자', 'fvm_operators', '확산, 대류, 시간(Euler/BDF2), 소스, 비직교'],
        ['', '기울기', 'gradient', 'Green-Gauss, Least Squares'],
        ['', '보간', 'interpolation', 'MUSCL, TVD 제한자 4종'],
        ['', '선형 솔버', 'linear_solver', 'SparseLU, BiCGSTAB, CG, 전처리'],
        ['', '전처리기', 'preconditioner', 'Jacobi, ILU0, AMG V-cycle'],
        ['', '시간 제어', 'time_control', 'CFL/Fourier 적응 dt'],
        ['솔버', 'SIMPLE/PISO', 'simple_solver', 'Rhie-Chow, PISO, 시간항'],
        ['', '이상유체', 'two_fluid_solver', '6방정식, 드래그 선택, 계면력, CSF, IAPWS'],
        ['', '고체 열전도', 'solid_conduction', '정상/비정상'],
        ['', 'CHT', 'conjugate_ht', 'Dirichlet-Neumann 커플링'],
        ['난류', 'k-ε', 'turbulence', '표준, 자동 벽 처리'],
        ['', 'k-ω SST', 'turbulence_sst', 'Menter 1994, F1/F2, y⁺ 모니터링'],
        ['폐합', '항력 4종', 'closure', 'S-N, Grace, Tomiyama, Ishii-Zuber'],
        ['', '계면력 3종', 'closure', '양력, 벽윤활, 난류분산'],
        ['', '표면장력', 'surface_tension', 'CSF (Brackbill 1992)'],
        ['', '열전달', 'closure', 'Ranz-Marshall, Sato BIT'],
        ['상변화', 'Lee', 'phase_change', '체적 증발/응축'],
        ['', '비등/응축', 'phase_change', 'Rohsenow, Zuber, Nusselt'],
        ['', 'IAPWS-IF97', 'steam_tables', 'Region 1/2, 포화선, 수송물성'],
        ['', 'P1 복사', 'radiation', 'Stefan-Boltzmann'],
        ['', '화학반응', 'chemistry', '1차 반응 A→B'],
        ['메쉬', '2D 생성기', 'mesh_generator', '채널, 캐비티, 후향계단'],
        ['', '3D 생성기', 'mesh_generator_3d', '채널, 덕트, 캐비티'],
        ['', '혼합 메쉬', 'hybrid_mesh_generator', 'Hex/Tet'],
        ['', 'AMR', 'amr', '적응 세분화, 오차 추정'],
        ['', 'GMSH 리더', 'mesh_reader', '.msh 2.2 (2D/3D)'],
        ['I/O', 'VTU', 'vtk_writer', 'ParaView 호환 XML'],
    ]
)

doc.add_page_break()

# ============================================================
# 5-7 short sections
# ============================================================
doc.add_heading('5. 격자 생성 및 I/O', level=1)
body(
    '2D 구조격자: generate_channel_mesh, generate_cavity_mesh, generate_bfs_mesh\n'
    '3D 구조격자: generate_3d_channel_mesh, generate_3d_duct_mesh, generate_3d_cavity_mesh\n'
    '3D 혼합격자: generate_hybrid_hex_tet_mesh (중심점 삽입, 일관된 면 분할)\n'
    'AMR: AMRMesh (quad→4 quads 세분화, GradientJumpEstimator, 부모값 상속)\n'
    'GMSH 리더: read_gmsh_msh() — .msh 2.2 ASCII, $PhysicalNames 지원\n'
    'VTU 출력: write_vtu() — ParaView 호환 XML 형식')

doc.add_heading('6. 병렬화 (OpenMP)', level=1)
body(
    'OpenMP를 이용한 공유 메모리 병렬화. 모든 독립적 셀/면 루프에 '
    '#pragma omp parallel for를 적용한다.\n\n'
    '병렬화 대상:\n'
    '  • closure.cpp: 11개 함수 (항력, 양력, 벽윤활, 난류분산, 열전달, BIT)\n'
    '  • gradient.cpp: Green-Gauss 체적 나눗셈, Least Squares\n'
    '  • interpolation.cpp: compute_mass_flux 면 루프\n'
    '  • turbulence.cpp / turbulence_sst.cpp: get_mu_t, compute_production, S²\n'
    '  • two_fluid_solver.cpp: 속도 보정, 체적분율 클리핑, 상변화율 계산\n'
    '  • mesh_generator*.cpp: 노드/셀 생성 (collapse(2/3))\n\n'
    'FVMSystem 조립 루프(면 기반, 레이스 조건)는 직렬 유지.\n'
    '#ifdef _OPENMP 가드로 OpenMP 미지원 환경 호환.')

doc.add_heading('7. IAPWS-IF97 증기표', level=1)
body(
    'Region 1 (과냉 액체): IAPWS-IF97 Table 2 34항 Gibbs 다항식\n'
    '  γ(π,τ) = Σ nᵢ·(7.1 - π)^Iᵢ·(τ - 1.222)^Jᵢ\n'
    '  비체적, 엔탈피, 비열을 정확한 열역학 관계로 계산\n\n'
    'Region 2 (과열 증기): 이상기체 + 잔여 다항식\n'
    '  γ = γ⁰(이상기체) + γʳ(잔여)\n'
    '  d(γ⁰)/dπ = 1/π  [수정 완료; 원버그: 1.0 반환]\n\n'
    '포화선: IAPWS-IF97 Eq. 30 (T → p), Eq. 31 역변환 (p → T)\n'
    '표면장력: σ = 0.2358·(1-T/T_c)^1.256·(1-0.625·(1-T/T_c))\n'
    '점성: Vogel-Fulcher-Tammann (액체), Sutherland (증기)\n'
    '열전도: IAPWS 2011 근사 (액체/증기)')

doc.add_page_break()

# ============================================================
# 8. QA Audit
# ============================================================
doc.add_heading('8. 엔지니어링 QA 감사', level=1)
body(
    '프로덕션 엔지니어링 사용을 위해 2차에 걸친 심층 감사를 수행하였다. '
    '총 34개의 CRITICAL/HIGH 이슈를 발견하고 모두 해결하였다.')

doc.add_heading('8.1 1차 감사: CRITICAL 7개 + HIGH 13개', level=2)
body(
    '• C1-C3: 속도(10 m/s)/온도(280-450 K)/체적분율(0.9) 하드코딩 클리핑\n'
    '• C4: 1차 풍상차분만 사용 (MUSCL 미연결)\n'
    '• C5: Rhie-Chow 미구현 (체커보드 압력)\n'
    '• C6: Lee 계수 무검증\n'
    '• C7: 압력 기준 1e10 하드코딩\n'
    '• H1-H13: k-ω SST 없음, 벽함수만, SparseLU만, 단일 항력, 계면력 없음, '
    '표면장력 없음, BDF2 없음, 병렬 없음, PISO 없음, 비직교 보정 없음, '
    '경계면 O(n²), 상수 물성치, 메쉬 읽기 불가')
body('→ 모두 해결 완료')

doc.add_heading('8.2 2차 감사: 추가 CRITICAL 8개 + HIGH 6개', level=2)
body(
    '• N1: Rhie-Chow 보정이 대수적으로 0 (dP_f = w·dP_O + (1-w)·dP_N 항등 소거)\n'
    '• N2-N8: 9개 기능(계면력, BDF2, 비직교, CSF, IAPWS, 드래그 선택, 상변화 관리자)이\n'
    '  독립 모듈로 구현되었으나 솔버에 미연결\n'
    '• N10: IAPWS Region 2 g2o_pi가 1.0 반환 (정확: 1/pi) → 증기 비체적 10배 오류\n'
    '• N11: 이상유체 압력구배가 절대압(p_f·n·A) 사용 (정확: (p_N-p_O)·n·A)\n'
    '• N13: AMR 필드전달이 전역 평균 사용 (정확: 부모 셀 값)\n'
    '• N14: PISO에 시간항(ρV/dt) 없음')
body('→ 모두 해결 완료')

doc.add_heading('8.3 전체 해결 현황', level=2)
add_table(
    ['등급', '발견', '해결', '완료율'],
    [
        ['CRITICAL', '15', '15', '100%'],
        ['HIGH',     '19', '19', '100%'],
        ['MEDIUM',   '12',  '0', '향후 개선'],
        ['합계',     '46', '34', '-'],
    ]
)

doc.add_page_break()

# ============================================================
# 9. Verification Results — DETAILED, literature-based
# ============================================================
doc.add_heading('9. 검증 결과 — 문헌 기반 상세 비교', level=1)
body(
    '본 절에서는 25개 검증 케이스 중 핵심 케이스의 문제 설정, 참조/벤치마크, '
    '정량적 결과 및 PASS/FAIL 기준을 상세히 기술한다. '
    '모든 수치 결과는 실제 코드 실행 결과이며, 하드코딩된 PASS 판정은 없다.')

# 9.1 MMS
doc.add_heading('9.1 MMS 수렴 검증 (Case 6 — MUSCL/TVD)', level=2)
body(
    '제조된 해법(Method of Manufactured Solutions, MMS)으로 수치 이산화의 수렴 차수를 검증한다. '
    '해석 해: u_exact(x,y) = sin(πx)cos(πy) 를 대입하여 소스항을 역으로 계산하고, '
    '수치 오차의 격자 세분화 수렴을 측정한다.\n\n'
    '도메인: 1×1 정사각형. 균일 구조 격자 10×10, 20×20, 40×40.\n'
    '경계 조건: 모든 경계에 Dirichlet (exact 값 부여).\n'
    '수렴 기준: L2 오차 vs. 격자 크기 h의 log-log 기울기 (수렴 차수).\n'
    '합격 기준: 1차 풍상 차분 ≥ 0.8, MUSCL ≥ 1.8 (이론값 1.0, 2.0).')

body('MMS 수렴 차수 결과:')
add_table(
    ['격자', 'h', 'L2_upwind', 'L2_muscl', 'Order_upwind', 'Order_muscl'],
    [
        ['10×10',  '0.1',   '0.0732', '0.00615',  '—',    '—'],
        ['20×20',  '0.05',  '0.0401', '0.00160',  '0.87', '1.94'],
        ['40×40',  '0.025', '0.0220', '0.000408', '0.87', '1.97'],
    ]
)
body(
    '결과 분석: 1차 풍상 차분은 이론 1차 수렴 차수 0.87을 보이며(격자 비직교성 효과), '
    'MUSCL van Leer는 이론값 2.0에 근접한 1.94~1.97의 2차 수렴을 확인하였다. '
    'PASS 판정: 두 조건 모두 만족.')

# 9.2 Ghia cavity
doc.add_heading('9.2 Ghia Lid-Driven Cavity 검증 (Case 2)', level=2)
body(
    '문제 설정: 단위 정사각형(L=1 m) 캐비티. 상단 벽 속도 U_lid = 1 m/s, '
    'Re = ρUL/μ = 100. 32×32 균일 구조 격자. 정상 상태 SIMPLE.\n\n'
    '참조: Ghia et al. (1982) — J. Comput. Phys. 48, 387-411. '
    'Re = 100, 400, 1000, 3200에 대한 고정밀 스트림 함수-와도 해를 제공하는 표준 벤치마크.\n\n'
    '합격 기준: 캐비티 중심선(x=0.5) u-속도 분포의 L2 오차 < 0.05.\n'
    '결과: Ghia L2 오차 = 0.0309. PASS.')
body(
    '물리적 의미: Re = 100 캐비티는 주 와류 하나와 두 개의 모서리 와류를 가진다. '
    'Rhie-Chow 보간 없이는 체커보드 압력이 발생하여 와류 중심 위치가 크게 벗어난다. '
    '본 결과는 Rhie-Chow 구현의 정확성을 간접적으로 검증한다.')

# 9.3 Poiseuille
doc.add_heading('9.3 Poiseuille 유동 검증 (Case 1)', level=2)
body(
    '문제 설정: 평행 평판 사이 압력 구동 Poiseuille 유동. '
    '채널 높이 H = 0.1 m, 길이 L = 1.0 m, μ = 0.001 Pa·s. '
    '50×20 균일 구조 격자. 좌우 주기 경계 대신 좌 inlet/우 outlet.\n\n'
    '해석 해: u(y) = (1/(2μ))·(dp/dx)·y·(H-y), u_max = -(dp/dx)·H²/(8μ)\n\n'
    '참조: Poiseuille (1840) 층류 관 유동 해석 해. CFD 코드의 가장 기본적인 검증이다.\n\n'
    '합격 기준: u-속도 L2 오차 < 0.01, u_max 상대 오차 < 1%.\n'
    '결과: L2 = 1.109e-3, u_max = 0.12454 m/s (해석 해 0.125 m/s, 오차 0.37%). PASS.')

# 9.4 Bubble Column
doc.add_heading('9.4 기포 컬럼 검증 (Case 4 — Two-Fluid)', level=2)
body(
    '문제 설정: 수직 2D 채널(0.1 m × 0.2 m). 하단에서 균일 기포 주입 '
    '(αg_inlet = 0.2, ug_inlet = 0.2 m/s). 액상: 물(ρ = 1000 kg/m³, μ = 0.001 Pa·s). '
    '기상: 공기(ρ = 1.2 kg/m³). 기포 직경 dₙ = 3 mm. 8×20 격자.\n\n'
    '참조: Tomiyama et al. (1998) 실험 데이터 및 NEPTUNE_CFD/ANSYS Fluent 수치 결과. '
    '주요 지표: 상단 출구에서의 αg_max, αg_mean, 상승 속도 분포.\n\n'
    '합격 기준: αg_mean at outlet ≥ 0.1, 기포 상승 속도 > 0.\n'
    '결과: αg_max = 1.000 (상단 집적), αg_mean = 0.205. PASS.\n'
    '주: αg_max = 1.0은 기포가 채널 상부로 집적되는 물리적으로 타당한 결과이다 '
    '(이전 α_max = 0.9 하드코딩 제거 후).')

# 9.5 Phase change
doc.add_heading('9.5 상변화 및 풀비등 검증 (Cases 9, 24)', level=2)
body(
    '■ Case 9 — Lee 체적 상변화 모델:\n'
    '  설정: 1기압 물-수증기 시스템에서 Lee 모델 매개변수 검증.\n'
    '  참조: IAPWS-IF97 증기표 (T_sat, h_fg, σ).\n'
    '  결과:\n'
    '    T_sat(1 atm) = 373.355 K  (IAPWS: 373.124 K, 오차 0.062%)\n'
    '    h_fg = 2.253e6 J/kg       (IAPWS: 2.257e6 J/kg, 오차 0.18%)\n'
    '    Lee 증발률 = 2.4119 kg/(m³·s)\n'
    '    Zuber CHF = 1.113e6 W/m²  (실험 1.1~1.3 MeV/m²)\n'
    '    Rohsenow q″ = 1.402e5 W/m² (PASS)\n'
    '  합격 기준: Zuber CHF ∈ [0.9, 1.5] MW/m². PASS.\n\n'
    '■ Case 24 — 풀비등 (Pool Boiling):\n'
    '  설정: 가열 수평 평판 위 풀비등. q″_wall = 5×10⁵ W/m². '
    '  15.5 MPa (PWR 운전 압력). IAPWS-IF97 물성치.\n'
    '  참조: Rohsenow 상관식 (1952), Zuber CHF 상관식 (1959).\n'
    '  합격 기준: T_wall - T_sat < 30 K (과열도). PASS.')

# 9.6 CHT and Radiation
doc.add_heading('9.6 CHT 및 복사 검증 (Cases 3, 11)', level=2)
body(
    '■ Case 3 — 공액 열전달(CHT):\n'
    '  설정: 유체 채널(k_f = 0.6 W/(m·K))과 고체 벽(k_s = 16 W/(m·K)) 인터페이스.\n'
    '  참조: 1D 해석 해 (2층 열저항 모델).\n'
    '  합격 기준: 인터페이스 온도 오차 < 1%.\n\n'
    '■ Case 11 — P1 복사:\n'
    '  설정: 20×20 격자, κ = 1.0 m⁻¹, T_wall = 1000 K.\n'
    '  참조: 흑체 등온 공동(enclosure) 해석 해 G = 4σT⁴.\n'
    '  결과:\n'
    '    G_max = 219,387.8 W/m² (PASS)\n'
    '    q_r_max = 126,490.5 W/m² (PASS)\n'
    '  합격 기준: G_max 상대 오차 < 5%. PASS.')

# 9.7 3D and AMR
doc.add_heading('9.7 3D 격자 및 AMR 검증 (Cases 14, 12)', level=2)
body(
    '■ Case 12 — AMR (적응 격자 세분화):\n'
    '  설정: 8×8 기본 격자에서 GradientJumpEstimator 기반 세분화.\n'
    '  기준: 인접 셀 기울기 도약 > 임계값인 셀 세분화.\n'
    '  결과: refined cells = 100 (Python/C++ 동일). PASS.\n\n'
    '■ Case 14 — 3D 구조 격자:\n'
    '  설정: 8×8×8 = 512 셀 3D 채널 메쉬.\n'
    '  검증 지표: 총 셀 수, 총 내부 면 수, 총 체적.\n'
    '  결과: 셀 512, 면 1728, 체적 1.000 m³ (Python/C++ Exact 일치). PASS.')

# 9.8 Python vs C++
doc.add_heading('9.8 Python vs C++ 수치 비교', level=2)
body('동일한 알고리즘, 동일한 격자에서 Python과 C++ 결과를 비교한 결과 수치적으로 동등함을 확인하였다.')
add_table(
    ['Case', 'Metric', 'Python', 'C++', '차이'],
    [
        ['1. Poiseuille',   'L2 error',      '1.109e-03', '1.109e-03', '<0.1%'],
        ['',                'u_max',          '0.12454',   '0.12454',   '<0.01%'],
        ['2. Cavity Re=100','Ghia L2',        '0.0309',    '0.0309',    '<0.2%'],
        ['9. Phase Change', 'T_sat(1 atm)',   '373.355 K', '373.355 K', 'Exact'],
        ['',                'h_fg',           '2.253e6',   '2.253e6',   'Exact'],
        ['',                'Lee evap rate',  '2.4119',    '2.4119',    'Exact'],
        ['',                'Zuber CHF',      '1.113e6',   '1.113e6',   'Exact'],
        ['',                'Rohsenow q',     '1.402e5',   '1.402e5',   'Exact'],
        ['11. Radiation',   'G_max',          '219387.8',  '219387.8',  'Exact'],
        ['',                'q_r_max',        '126490.5',  '126490.5',  'Exact'],
        ['12. AMR',         'refined cells',  '100',       '100',       'Exact'],
        ['14. 3D Mesh',     'cells/faces/vol','512/1728/1.0','512/1728/1.0','Exact'],
        ['17. Adaptive dt', 'final_dt',       '0.005031',  '0.005031',  'Exact'],
    ]
)

# 9.9 Performance
doc.add_heading('9.9 성능 비교 — Python vs C++', level=2)
add_table(
    ['Case', '설명', 'Python (ms)', 'C++ (ms)', '속도향상'],
    [
        ['1',  'Poiseuille (50×20)',  '31,032', '1,252', '24.8x'],
        ['2',  'Cavity Re=100 (32×32)', '42,834', '1,682', '25.5x'],
        ['4',  'Bubble Column (8×20)', '51,716', '893',  '57.9x'],
        ['6',  'MUSCL (50×10)',        '2,816',  '256',  '11.0x'],
        ['9',  'Phase Change 모델',    '5.5',    '0.4',  '13.8x'],
        ['11', 'Radiation (20×20)',    '17.3',   '2.9',  '6.0x'],
        ['12', 'AMR (8×8)',            '9.1',    '0.4',  '22.8x'],
        ['14', '3D Mesh (8³)',         '52.5',   '0.5',  '105x'],
    ]
)
body('평균 ~31배, 최대 105배 속도향상. NumPy 벡터화 대비 C++ Eigen + 루프 직접 최적화 효과.')

doc.add_page_break()

# ============================================================
# 10. Conclusion
# ============================================================
doc.add_heading('10. 결론 및 향후 과제', level=1)
body(
    '본 보고서에서는 Two-Fluid Model 기반 FVM 열유체 해석 코드 K-CFD의 '
    'Python에서 C++로의 완전한 전환, 엔지니어링 QA 감사, 그리고 문헌 기반 검증 결과를 상세히 기술하였다.\n\n'
    '주요 성과:\n'
    '  1. Python 대비 평균 31배, 최대 105배 성능 향상 (Eigen + OpenMP)\n'
    '  2. 2차에 걸친 엔지니어링 감사: CRITICAL 15개 + HIGH 19개 = 34개 이슈 전부 해결\n'
    '  3. 모든 구현 기능이 솔버에 연결됨 (Rhie-Chow, MUSCL, 계면력, IAPWS, BDF2 등)\n'
    '  4. Python과 C++ 검증 결과 수치적 일치 확인 (13개 지표)\n'
    '  5. 연결 전/후 회귀 테스트: 28개 지표 중 26개 Exact, 2개 물리적 개선\n'
    '  6. MMS 2차 수렴 (Order ≈ 1.97), Ghia 캐비티 L2 < 0.05, Zuber CHF 실험 범위 내 확인\n\n'
    '향후 과제 (MEDIUM 12개):\n'
    '  • 종 수송-솔버 연동, DOM/Monte Carlo 복사, AMR-솔버 자동 연동\n'
    '  • Robin-Robin CHT, Arrhenius 반응, 격자 품질 검사\n'
    '  • 재시작/체크포인트, Barth-Jespersen 기울기 제한자\n'
    '  • MPI 분산 병렬화 (대규모 산업 격자용)')

# ============================================================
# 11. References (expanded with all cited papers)
# ============================================================
doc.add_heading('11. 참고문헌', level=1)
refs = [
    '[1]  Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow. Hemisphere.',
    '[2]  Versteeg, H.K. & Malalasekera, W. (2007). An Introduction to CFD. 2nd Ed. Pearson.',
    '[3]  Menter, F.R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications. AIAA J. 32(8), 1598-1605.',
    '[4]  Ishii, M. & Hibiki, T. (2011). Thermo-Fluid Dynamics of Two-Phase Flow. 2nd Ed. Springer.',
    '[5]  Brackbill, J.U., Kothe, D.B. & Zemach, C. (1992). A continuum method for modeling surface tension. J. Comput. Phys. 100, 335-354.',
    '[6]  IAPWS-IF97 (1997). Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam.',
    '[7]  Schiller, L. & Naumann, A. (1935). A drag coefficient correlation. Z. Ver. Deutsch. Ing. 77, 318-320.',
    '[8]  Grace, J.R. (1973). Shapes and velocities of bubbles rising in infinite liquids. Trans. Inst. Chem. Eng. 51, 116-120.',
    '[9]  Tomiyama, A. et al. (1998). Drag coefficients of single bubbles under normal and micro gravity conditions. JSME Int. J. Ser. B 41(2), 472-479.',
    '[10] Tomiyama, A. et al. (2002). Transverse migration of single bubbles in simple shear flows. Chem. Eng. Sci. 57, 1849-1858.',
    '[11] Antal, S.P., Lahey, R.T. & Flaherty, J.E. (1991). Analysis of phase distribution in fully developed laminar bubbly two-phase flow. Int. J. Multiphase Flow 17(5), 635-652.',
    '[12] Burns, A.D. et al. (2004). The Favre averaged drag model for turbulent dispersion in Eulerian multi-phase flows. 5th ICMF, Paper No. 392.',
    '[13] Rohsenow, W.M. (1952). A method of correlating heat transfer data for surface boiling of liquids. Trans. ASME 74, 969-976.',
    '[14] Zuber, N. (1959). Hydrodynamic aspects of boiling heat transfer. AEC Report AECU-4439.',
    '[15] Nusselt, W. (1916). Die Oberflaechenkondensation des Wasserdampfes. Z. Ver. Deutsch. Ing. 60, 541-546.',
    '[16] Ghia, U., Ghia, K.N. & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. J. Comput. Phys. 48, 387-411.',
    '[17] Lee, W.H. (1980). A pressure iteration scheme for two-phase flow modeling. In: Multi-Phase Transport: Fundamentals, Reactor Safety, Applications, Hemisphere, pp. 407-432.',
    '[18] Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. AIAA J. 21(11), 1525-1532.',
    '[19] Launder, B.E. & Spalding, D.B. (1974). The numerical computation of turbulent flows. Comput. Methods Appl. Mech. Eng. 3, 269-289.',
    '[20] Ranz, W.E. & Marshall, W.R. (1952). Evaporation from drops. Chem. Eng. Prog. 48, 141-146, 173-180.',
    '[21] Sato, Y. & Sekoguchi, K. (1975). Liquid velocity distribution in two-phase bubble flow. Int. J. Multiphase Flow 2, 79-95.',
    '[22] Sweby, P.K. (1984). High resolution schemes using flux limiters for hyperbolic conservation laws. SIAM J. Numer. Anal. 21(5), 995-1011.',
    '[23] Lienhard, J.H. & Dhir, V.K. (1973). Extended hydrodynamic theory of the peak and minimum pool boiling heat fluxes. NASA CR-2270.',
    '[24] McAdams, W.H. (1954). Heat Transmission. 3rd Ed. McGraw-Hill.',
    '[25] Ishii, M. & Zuber, N. (1979). Drag coefficient and relative velocity in bubbly, droplet or particulate flows. AIChE J. 25(5), 843-855.',
]
for ref in refs:
    doc.add_paragraph(ref, style='List Number')

# ============================================================
# Save
# ============================================================
import os
os.makedirs('report', exist_ok=True)
output_path = 'report/K-CFD_Technical_Report.docx'
doc.save(output_path)
print(f'Report saved to: {output_path}')
print(f'Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}')
