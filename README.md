# Two-Fluid FVM Solver (C++)

Euler-Euler 이상유체 유한체적법(FVM) 솔버. 원자력 열수력, 화학공정, 다상유동 해석용.

## 빌드

```bash
cd cpp/build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j4
```

**의존성**: Eigen 3.4+, C++17, CMake 3.16+
**선택**: pybind11 (Python 바인딩)

## 실행

```bash
# 단상 유동
./twofluid_solver --case channel --nx 100 --ny 50 --Re 100
./twofluid_solver --case cavity --nx 64 --Re 1000
./twofluid_solver --case bfs --nx 120 --ny 40 --Re 5100 --turbulence

# 이상유동
./twofluid_solver --case bubble --nx 8 --ny 20
```

## 모듈 구성 (22개)

| 카테고리 | 모듈 | 파일 |
|---------|------|------|
| **Core** | 메쉬, 필드, FVM 연산자 | `mesh`, `fields`, `fvm_operators` |
| | 기울기 재구성 | `gradient` (Green-Gauss, Least Squares) |
| | 면 보간, TVD 제한자 | `interpolation` (MUSCL, minmod, van Leer, superbee, van Albada) |
| | 선형 솔버 | `linear_solver` (SparseLU, BiCGSTAB) |
| | 전처리기 | `preconditioner` (Jacobi, ILU0) |
| | 적응 시간 제어 | `time_control` (CFL/Fourier 기반) |
| **솔버** | 단상 SIMPLE | `simple_solver` |
| | 이상유체 (6방정식) | `two_fluid_solver` |
| | 고체 열전도 | `solid_conduction` |
| | 공액 열전달 | `conjugate_ht` |
| **물리 모델** | k-ε 난류 | `turbulence` |
| | 계면 폐합 관계 | `closure` (Schiller-Naumann 항력, Ranz-Marshall, Sato BIT) |
| | 상변화 | `phase_change` (Lee, Rohsenow, Zuber CHF, Nusselt 응축) |
| | P1 복사 | `radiation` |
| | 화학 반응 | `chemistry` (1차 반응 A→B) |
| **메쉬** | 2D 생성기 | `mesh_generator` (채널, 캐비티, 후향계단) |
| | 3D 생성기 | `mesh_generator_3d` (채널, 덕트, 캐비티) |
| | 혼합 메쉬 | `hybrid_mesh_generator` (Hex/Tet) |
| | 적응 격자 세분화 | `amr` (AMRMesh, 오차추정) |
| **I/O** | VTU 출력 | `vtk_writer` |
| | Python 바인딩 | `bindings` (pybind11) |

## 테스트

```bash
./test_basic          # 메쉬, 필드, 기울기 (5 tests)
./test_simple         # 선형솔버, 보간, Poiseuille (3 tests)
./test_new_modules    # 전처리기, 시간제어, 상변화, 고체, 복사, AMR, 3D (7 tests)
./verification_cases  # 10개 검증 케이스 (Python 결과와 대조 검증 완료)
```

---

## 엔지니어링 감사 결과

> **현재 상태**: 프로토타입/검증용. 프로덕션 사용을 위해 아래 이슈 해결 필요.

### CRITICAL — 프로덕션에서 잘못된 결과를 생성

| ID | 문제 | 위치 | 상태 |
|----|------|------|------|
| C1 | **속도 하드코딩 클리핑 (10 m/s)** — 증기관, 노즐 등 고속 유동 불가 | `two_fluid_solver.cpp` | TODO |
| C2 | **온도 하드코딩 클리핑 [280, 450] K** — PWR(573K), 극저온, 용융염 불가 | `two_fluid_solver.cpp` | TODO |
| C3 | **체적분율 상한 0.9 고정** — LOCA, dry-out, 분리류 불가 | `two_fluid_solver.cpp` | TODO |
| C4 | **1차 풍상차분만 사용** — MUSCL 존재하나 미연결. 과도한 수치확산 | `fvm_operators.cpp` | TODO |
| C5 | **Rhie-Chow 보간 미구현** — 체커보드 압력 진동, 모든 압력장 신뢰불가 | `simple_solver.cpp`, `two_fluid_solver.cpp` | TODO |
| C6 | **Lee 모델 계수 검증 없음** — 0.001~100+ 범위, 경고 없이 오류 전파 | `phase_change.cpp` | TODO |
| C7 | **압력 기준점 1e10 대각 고정** — 조건수 파괴, 스케일 무관 하드코딩 | `two_fluid_solver.cpp` | TODO |

### HIGH — 정확도/견고성의 중대한 제한

| ID | 문제 | 상태 |
|----|------|------|
| H1 | **표준 k-ε만 구현** — 분리류 과소예측. k-ω SST 필요 | TODO |
| H2 | **벽함수만 (high-Re)** — y+ 모니터링 없음, Low-Re 모델 없음 | TODO |
| H3 | **SparseLU 직접 솔버** — >100k셀 메모리 부족. AMG 필요 | TODO |
| H4 | **Schiller-Naumann 항력만** — Grace, Tomiyama, Ishii-Zuber 미구현 | TODO |
| H5 | **계면력: 항력만** — 양력, 벽윤활력, 난류분산력 없음 | TODO |
| H6 | **표면장력 없음** — CSF, 모세관 압력 미구현 | TODO |
| H7 | **후방 오일러만 (1차 시간적분)** — BDF2, Crank-Nicolson 없음 | TODO |
| H8 | **병렬 컴퓨팅 없음** — MPI/OpenMP 전무 | TODO |
| H9 | **SIMPLE만** — SIMPLEC, PISO 미구현 | TODO |
| H10 | **비직교 보정 없음** (fvm_operators 내) | TODO |
| H11 | **경계면 탐색 O(n²)** — 대규모 격자 성능 붕괴 | TODO |
| H12 | **물성치 상수 고정** — 온도/압력 의존 없음. IAPWS-IF97 필요 | TODO |
| H13 | **메쉬 파일 읽기 불가** — CGNS, Fluent .msh, GMSH 리더 없음 | TODO |

### MEDIUM — 일부 경우 허용 가능하나 범용성 제한

| ID | 문제 | 상태 |
|----|------|------|
| M1 | 종 수송이 이상유체 솔버와 미연동 | TODO |
| M2 | P1 복사만 (DOM, Monte Carlo 없음) | TODO |
| M3 | AMR이 솔버와 미연동 (2D만, 거칠게하기 없음) | TODO |
| M4 | CHT 커플링 단순 (Dirichlet-Neumann만) | TODO |
| M5 | 1차 반응만 (Arrhenius 없음) | TODO |
| M6 | 격자 품질 검사 없음 | TODO |
| M7 | 재시작/체크포인트 없음 | TODO |
| M8 | 기울기 제한자 없음 (Barth-Jespersen) | TODO |
| M9 | 적응 시간 제어가 솔버에 미연동 | TODO |
| M10 | 기체상 벽 BC가 자유 미끄럼만 | TODO |
| M11 | CMakeLists.txt에 하드코딩된 Eigen 경로 | TODO |
| M12 | MinGW 최적화 우회 (잠재적 정확도 문제) | TODO |

---

## 개선 로드맵

### Phase 1 — Critical 해소 (1-2주)

1. 하드코딩 클리핑 제거 → 사용자 설정 한계값 + 발산 탐지 (C1, C2, C3)
2. Rhie-Chow 운동량 보간 구현 (C5)
3. MUSCL 2차 대류를 기본값으로 연결 (C4)
4. 압력 기준점 스케일링 (C7)
5. 경계면 해시맵 캐시 (H11)
6. 기존 전처리기를 솔버에 연결 (H3 부분)

### Phase 2 — 핵심 물리 보강 (1-3개월)

1. k-ω SST 난류 모델 (H1)
2. IAPWS-IF97 물성치 테이블 (H12)
3. 양력/벽윤활력/난류분산력 (H5)
4. BDF2 시간 적분 (H7)
5. PISO 알고리즘 (H9)
6. Grace/Tomiyama 항력 (H4)
7. 비직교 보정 (H10)
8. 기울기 제한자 (M8)

### Phase 3 — 확장성 확보 (3-6개월)

1. AMG 선형 솔버 (AMGCL 연동) (H3)
2. OpenMP 스레딩 (H8)
3. 메쉬 파일 리더 (GMSH, CGNS) (H13)
4. 재시작/체크포인트 (M7)
5. 격자 품질 검사 (M6)

### Phase 4 — 프로덕션급 전환 (6-12개월)

1. MPI 도메인 분할 (H8)
2. 표면장력 CSF 모델 (H6)
3. Low-Re 벽 처리 (H2)
4. 다성분 종 수송 + Arrhenius (M1, M5)
5. DOM 복사 (M2)
6. 3D AMR + 솔버 연동 (M3)

---

## 라이선스

MIT License
