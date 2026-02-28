# Two-Fluid FVM 열유체 솔버

Python 기반 Two-Fluid Model (Euler-Euler) 유한체적법 솔버. 25개 검증 케이스를 포함하며 DOCX 보고서를 자동 생성한다.

GitHub: [https://github.com/senily0/K-CFD](https://github.com/senily0/K-CFD)

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [주요 기능](#주요-기능)
3. [디렉토리 구조](#디렉토리-구조)
4. [설치](#설치)
5. [실행](#실행)
6. [검증 케이스](#검증-케이스)
7. [보고서 생성](#보고서-생성)
8. [참고문헌](#참고문헌)

---

## 프로젝트 개요

이 솔버는 기-액 이상유동(two-phase flow)을 Euler-Euler 프레임워크로 수치 해석한다. 두 상(phase)을 각각 연속체로 취급하며, 6-equation 모델(각 상별 연속, 운동량, 에너지 방정식)을 유한체적법(FVM)으로 이산화한다. 속도-압력 커플링에는 SIMPLE 알고리즘을 사용한다.

**핵심 수치 기법:**

- 유한체적법(FVM) 이산화 — 확산, 대류, 소스, 시간항
- SIMPLE 알고리즘(Patankar & Spalding, 1972)으로 속도-압력 커플링
- BiCGSTAB, 직접법, AMG 선형 솔버
- ILU / Jacobi / AMG 전처리기
- MUSCL / TVD 고차 공간 스킴
- CFL 기반 적응 시간 보폭

---

## 주요 기능

### 핵심 솔버

| 모듈 | 내용 |
|------|------|
| `core/fvm_operators.py` | FVM 이산화 연산자 (확산, 대류, 소스, 시간항) |
| `core/fields.py` | 스칼라/벡터 필드 자료구조 |
| `core/gradient.py` | 그래디언트 재구성 |
| `core/linear_solver.py` | BiCGSTAB, 직접법, AMG 솔버 |
| `core/preconditioner.py` | ILU, Jacobi, AMG 전처리기 |
| `core/interpolation.py` | 셀 중심 → 면 보간 |
| `core/time_control.py` | CFL 기반 적응 시간 보폭 |
| `core/gpu_solver.py` | CuPy GPU 가속 솔버 |

### 물리 모델

| 모듈 | 내용 |
|------|------|
| `models/two_fluid.py` | Two-Fluid Model (Euler-Euler), 6-equation |
| `models/single_phase.py` | 단상 SIMPLE 솔버 |
| `models/phase_change.py` | 상변화 — Lee 모델 (증발/응축) |
| `models/radiation.py` | P1 복사 모델 |
| `models/chemistry.py` | 화학반응 (1차 반응) |
| `models/turbulence.py` | k-epsilon 난류 모델 |
| `models/conjugate_ht.py` | 공액열전달(CHT) — 고체/유체 결합 |
| `models/closure.py` | 항력, 양력 등 계면 클로저 상관식 |

### 격자

| 모듈 | 내용 |
|------|------|
| `mesh/mesh_generator.py` | 2D 정렬 격자 생성 |
| `mesh/mesh_generator_3d.py` | 3D 정렬 격자 생성 |
| `mesh/hybrid_mesh_generator.py` | Hex/Tet 혼합 격자 생성 |
| `mesh/mesh_reader.py` | FVM 격자 읽기/파싱 |
| `mesh/amr.py` | 적응 격자 세분화(AMR) |
| `mesh/vtk_exporter.py` | VTK/VTU 포맷 내보내기 |

### 병렬화 및 후처리

| 모듈 | 내용 |
|------|------|
| `parallel/partitioning.py` | MPI 영역 분할 |
| `parallel/mpi_solver.py` | 병렬 선형 솔버 |
| `visualization/vtu_renderer.py` | VTU 렌더러 (meshio + matplotlib) |
| `visualization/paraview_render.py` | ParaView 스타일 렌더링 |
| `visualization/web_dashboard.py` | Flask 실시간 웹 대시보드 |
| `report/report_generator.py` | DOCX 보고서 자동 생성 |

---

## 디렉토리 구조

```
twofluid_fvm/
├── core/               # FVM 핵심 (필드, 연산자, 솔버, GPU, 전처리기)
├── mesh/               # 격자 생성/읽기, AMR, 혼합격자, VTK export
├── models/             # Two-Fluid, 단상, 상변화, 복사, 화학반응, 난류
├── parallel/           # MPI 분할, 병렬 솔버
├── verification/       # 25개 검증 케이스
├── visualization/      # VTU 렌더러, ParaView, 웹 대시보드
├── figures/            # 검증 그래프 및 메쉬 시각화 그림
├── report/             # 보고서 생성기 및 DOCX 보고서
├── results/            # 케이스 실행 결과 (pickle)
├── generate_report.py  # 25개 케이스 실행 + DOCX 생성 (권장)
└── main.py             # 대안 실행 진입점
```

---

## 설치

Python 3.8 이상 필요.

```bash
pip install numpy scipy matplotlib python-docx flask meshio
```

GPU 가속을 사용하려면 CUDA 환경에서 CuPy를 추가로 설치한다.

```bash
pip install cupy-cuda12x  # CUDA 버전에 맞게 선택
```

---

## 실행

### 전체 케이스 실행 + 보고서 생성 (권장)

```bash
python generate_report.py
```

25개 케이스를 순차 실행하고, VTU 렌더링 후 DOCX 보고서를 생성한다. 각 케이스에는 타임아웃이 설정되어 있으며, 타임아웃 발생 시 이전 캐시 결과를 사용한다.

### 개별 케이스 실행

```bash
python -m verification.case1_poiseuille
python -m verification.case4_bubble_column
python -m verification.case20_edwards_blowdown
```

### main.py 사용

```bash
python main.py
```

---

## 검증 케이스

| 케이스 | 이름 | 설명 |
|--------|------|------|
| 1 | Poiseuille | 2D 층류 Poiseuille 유동 (해석해 비교) |
| 2 | Lid-Driven Cavity | Re=100, 400 cavity 유동 |
| 3 | CHT | 공액열전달 (구리-물 고체/유체 결합) |
| 4 | Bubble Column | 기포탑 이상유동 (Euler-Euler) |
| 5 | 3D Duct | 3D 덕트 Poiseuille 유동 |
| 6 | MUSCL/TVD | 고차 공간 스킴 비교 |
| 7 | Unstructured | 삼각형 비정렬 격자 검증 |
| 8 | MPI Parallel | MPI 영역 분할 병렬화 |
| 9 | Stefan Problem | Lee 모델 상변화 (계면 추적) |
| 10 | Chemical Reaction | 1차 반응 plug flow reactor |
| 11 | P1 Radiation | P1 복사 모델 (1D slab) |
| 12 | AMR | 적응 격자 세분화 |
| 13 | GPU Acceleration | CuPy GPU 벤치마크 |
| 14 | 3D Cavity | 3D lid-driven cavity (Re=100) |
| 15 | 3D Natural Convection | 3D 자연대류 (Boussinesq) |
| 16 | Preconditioner | ILU / Jacobi / AMG 전처리기 비교 |
| 17 | Adaptive dt | CFL 기반 적응 시간 보폭 |
| 18 | Web Dashboard | Flask 실시간 시각화 |
| 19 | Boiling/Condensation | Rohsenow / Nusselt 상관식 검증 |
| 20 | Edwards Blowdown | 파이프 블로우다운 flashing (실험 비교) |
| 21 | 6-Eq Heated Channel | 6-equation 가열 채널 비평형 상변화 |
| 22 | Hybrid Mesh | Hex/Tet 혼합 격자 |
| 23 | 3D Transient Channel | 3D 과도 채널 유동 |
| 24 | Pool Boiling | 풀 비등 시뮬레이션 |
| 25 | Film Condensation | 막 응축 (Nusselt 이론 비교) |

---

## 보고서 생성

`generate_report.py` 실행 시 다음 순서로 진행된다.

1. 25개 검증 케이스 실행 (케이스별 타임아웃 적용, 캐시 병합)
2. `visualization/vtu_renderer.py`로 메쉬 시각화 그림 렌더링
3. `report/report_generator.py`로 DOCX 보고서 생성

출력 파일: `report/TwoFluid_FVM_Report.docx`

보고서에는 25개 케이스 결과와 55개 이상의 검증 그래프가 포함된다.

---

## 참고문헌

- Patankar, S.V. & Spalding, D.B. (1972). *A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows.* Int. J. Heat Mass Transfer, 15, 1787-1806. — SIMPLE 알고리즘
- Edwards, A.R. & O'Brien, T.P. (1970). *Studies of phenomena connected with the depressurization of water reactors.* J. Br. Nucl. Energy Soc., 9, 125-135. — 파이프 블로우다운 실험 (케이스 20)
- Rohsenow, W.M. (1952). *A method of correlating heat transfer data for surface boiling of liquids.* Trans. ASME, 74, 969-976. — 핵비등 상관식 (케이스 19, 24)
- Nusselt, W. (1916). *Die Oberflachenkondensation des Wasserdampfes.* Z. VDI, 60, 541-546. — 막 응축 이론 (케이스 25)
- Lee, W.H. (1980). *A pressure iteration scheme for two-phase flow modeling.* — 상변화 소스항 모델 (케이스 9, 21)
