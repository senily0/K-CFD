"""
전체 검증 케이스 실행 스크립트.

22개의 검증 케이스를 순서대로 실행하고 결과를 종합한다.
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.case1_poiseuille import run_case1
from verification.case2_cavity import run_case2
from verification.case3_cht import run_case3
from verification.case4_bubble_column import run_case4
from verification.case5_3d_duct import run_case5
from verification.case6_muscl import run_case6
from verification.case7_unstructured import run_case7
from verification.case8_mpi import run_case8
from verification.case9_phase_change import run_case9
from verification.case10_reaction import run_case10
from verification.case11_radiation import run_case11
from verification.case12_amr import run_case12
from verification.case13_gpu import run_case13
from verification.case14_3d_cavity import run_case14
from verification.case15_3d_convection import run_case15
from verification.case16_preconditioner import run_case16
from verification.case17_adaptive_dt import run_case17
from verification.case18_web_dashboard import run_case18
from verification.case19_boiling_condensation import run_case19
from verification.case20_edwards_blowdown import run_case20
from verification.case21_6eq_heated_channel import run_case21
from verification.case22_hybrid_mesh import run_case22
from verification.case23_3d_transient import run_case23
from verification.case24_pool_boiling import run_case24
from verification.case25_film_condensation import run_case25


def _run_case(name, func, results_dir, figures_dir, **kwargs):
    """단일 케이스 실행 래퍼."""
    try:
        t0 = time.time()
        result = func(results_dir=results_dir, figures_dir=figures_dir, **kwargs)
        if isinstance(result, dict):
            result['elapsed_time'] = time.time() - t0
        else:
            result = {'raw': result, 'elapsed_time': time.time() - t0}
        print(f"  {name} 완료: {result['elapsed_time']:.1f}초")
        return result
    except Exception as e:
        print(f"  {name} 실패: {e}")
        traceback.print_exc()
        return {'error': str(e)}


def run_all_cases(results_dir: str = "results",
                  figures_dir: str = "figures") -> dict:
    """
    전체 검증 케이스 실행.

    Returns
    -------
    all_results : {case_name: result_dict}
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    all_results = {}
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  전체 검증 실행 시작 (25 Cases)")
    print("=" * 70)

    # Case 1: Poiseuille
    all_results['case1_poiseuille'] = _run_case(
        'Case 1', run_case1, results_dir, figures_dir)

    # Case 2: Cavity
    try:
        t0 = time.time()
        result2 = run_case2(re_list=[100, 400], n_grid=32,
                           results_dir=results_dir, figures_dir=figures_dir)
        elapsed = time.time() - t0
        all_results['case2_cavity'] = {
            'results': result2, 'elapsed_time': elapsed}
        print(f"  Case 2 완료: {elapsed:.1f}초")
    except Exception as e:
        print(f"  Case 2 실패: {e}")
        all_results['case2_cavity'] = {'error': str(e)}

    # Case 3: CHT
    all_results['case3_cht'] = _run_case(
        'Case 3', run_case3, results_dir, figures_dir)

    # Case 4: Bubble Column
    all_results['case4_bubble_column'] = _run_case(
        'Case 4', run_case4, results_dir, figures_dir)

    # Case 5: 3D Duct Poiseuille
    all_results['case5_3d_duct'] = _run_case(
        'Case 5', run_case5, results_dir, figures_dir)

    # Case 6: MUSCL/TVD
    all_results['case6_muscl'] = _run_case(
        'Case 6', run_case6, results_dir, figures_dir)

    # Case 7: Unstructured (Triangle)
    all_results['case7_unstructured'] = _run_case(
        'Case 7', run_case7, results_dir, figures_dir)

    # Case 8: MPI Parallel
    all_results['case8_mpi'] = _run_case(
        'Case 8', run_case8, results_dir, figures_dir)

    # Case 9: Phase Change
    all_results['case9_phase_change'] = _run_case(
        'Case 9', run_case9, results_dir, figures_dir)

    # Case 10: Reaction
    all_results['case10_reaction'] = _run_case(
        'Case 10', run_case10, results_dir, figures_dir)

    # Case 11: Radiation
    all_results['case11_radiation'] = _run_case(
        'Case 11', run_case11, results_dir, figures_dir)

    # Case 12: AMR
    all_results['case12_amr'] = _run_case(
        'Case 12', run_case12, results_dir, figures_dir)

    # Case 13: GPU
    all_results['case13_gpu'] = _run_case(
        'Case 13', run_case13, results_dir, figures_dir)

    # Case 14: 3D Lid-Driven Cavity
    all_results['case14_3d_cavity'] = _run_case(
        'Case 14', run_case14, results_dir, figures_dir)

    # Case 15: 3D Natural Convection
    all_results['case15_3d_convection'] = _run_case(
        'Case 15', run_case15, results_dir, figures_dir)

    # Case 16: Preconditioner
    all_results['case16_preconditioner'] = _run_case(
        'Case 16', run_case16, results_dir, figures_dir)

    # Case 17: Adaptive Time Stepping
    all_results['case17_adaptive_dt'] = _run_case(
        'Case 17', run_case17, results_dir, figures_dir)

    # Case 18: Web Dashboard
    all_results['case18_web_dashboard'] = _run_case(
        'Case 18', run_case18, results_dir, figures_dir)

    # Case 19: Boiling/Condensation
    all_results['case19_boiling_condensation'] = _run_case(
        'Case 19', run_case19, results_dir, figures_dir)

    # Case 20: Edwards Pipe Blowdown (Flashing)
    all_results['case20_edwards_blowdown'] = _run_case(
        'Case 20', run_case20, results_dir, figures_dir)

    # Case 21: 6-Equation 가열 채널
    all_results['case21_6eq_heated_channel'] = _run_case(
        'Case 21', run_case21, results_dir, figures_dir)

    # Case 22: 혼합 격자(Hex/Tet) Poiseuille
    all_results['case22_hybrid_mesh'] = _run_case(
        'Case 22', run_case22, results_dir, figures_dir)

    # Case 23: 3D 과도 채널 유동
    all_results['case23_3d_transient'] = _run_case(
        'Case 23', run_case23, results_dir, figures_dir)

    # Case 24: 풀 비등
    all_results['case24_pool_boiling'] = _run_case(
        'Case 24', run_case24, results_dir, figures_dir)

    # Case 25: 막 응축
    all_results['case25_film_condensation'] = _run_case(
        'Case 25', run_case25, results_dir, figures_dir)

    total_elapsed = time.time() - total_start

    # 결과 종합
    print("\n" + "=" * 70)
    print("  검증 결과 종합")
    print("=" * 70)

    # Case 1
    r1 = all_results.get('case1_poiseuille', {})
    if 'error' not in r1:
        status = "PASS" if r1.get('L2_error', 1) < 0.01 else "FAIL"
        print(f"  Case  1 (Poiseuille):     {status} (L2={r1.get('L2_error', 'N/A'):.4e})")
    else:
        print(f"  Case  1 (Poiseuille):     ERROR - {r1['error']}")

    # Case 2
    r2 = all_results.get('case2_cavity', {})
    if 'error' not in r2:
        for Re, res in r2.get('results', {}).items():
            status = "PASS" if res.get('converged', False) else "FAIL"
            print(f"  Case  2 (Cavity Re={Re}):  {status}")
    else:
        print(f"  Case  2 (Cavity):         ERROR - {r2['error']}")

    # Case 3
    r3 = all_results.get('case3_cht', {})
    if 'error' not in r3:
        status = "PASS" if r3.get('converged', False) else "FAIL"
        print(f"  Case  3 (CHT):            {status}")
    else:
        print(f"  Case  3 (CHT):            ERROR - {r3['error']}")

    # Case 4
    r4 = all_results.get('case4_bubble_column', {})
    if 'error' not in r4:
        status = "PASS" if r4.get('physical_validity', False) else "FAIL"
        print(f"  Case  4 (Bubble Column):  {status}")
    else:
        print(f"  Case  4 (Bubble Column):  ERROR - {r4['error']}")

    # Case 5
    r5 = all_results.get('case5_3d_duct', {})
    if 'error' not in r5:
        status = "PASS" if r5.get('L2_error', 1) < 0.05 else "FAIL"
        print(f"  Case  5 (3D Duct):        {status} (L2={r5.get('L2_error', 'N/A'):.4e})")
    else:
        print(f"  Case  5 (3D Duct):        ERROR - {r5['error']}")

    # Case 6
    r6 = all_results.get('case6_muscl', {})
    if 'error' not in r6:
        status = "PASS" if r6.get('muscl_sharper', r6.get('converged', False)) else "FAIL"
        print(f"  Case  6 (MUSCL):          {status}")
    else:
        print(f"  Case  6 (MUSCL):          ERROR - {r6['error']}")

    # Case 7
    r7 = all_results.get('case7_unstructured', {})
    if 'error' not in r7:
        status = "PASS" if r7.get('L2_error', 1) < 0.10 else "FAIL"
        print(f"  Case  7 (Triangle Mesh):  {status} (L2={r7.get('L2_error', 'N/A'):.4e})")
    else:
        print(f"  Case  7 (Triangle Mesh):  ERROR - {r7['error']}")

    # Case 8
    r8 = all_results.get('case8_mpi', {})
    if 'error' not in r8:
        status = "PASS" if r8.get('converged', False) else "FAIL"
        print(f"  Case  8 (MPI Parallel):   {status}")
    else:
        print(f"  Case  8 (MPI Parallel):   ERROR - {r8['error']}")

    # Case 9
    r9 = all_results.get('case9_phase_change', {})
    if 'error' not in r9:
        status = "PASS" if r9.get('converged', False) else "FAIL"
        print(f"  Case  9 (Phase Change):   {status}")
    else:
        print(f"  Case  9 (Phase Change):   ERROR - {r9['error']}")

    # Case 10
    r10 = all_results.get('case10_reaction', {})
    if 'error' not in r10:
        status = "PASS" if r10.get('L2_error', 1) < 0.05 else "FAIL"
        print(f"  Case 10 (Reaction):       {status} (L2={r10.get('L2_error', 'N/A'):.4e})")
    else:
        print(f"  Case 10 (Reaction):       ERROR - {r10['error']}")

    # Case 11
    r11 = all_results.get('case11_radiation', {})
    if 'error' not in r11:
        status = "PASS" if r11.get('converged', False) else "FAIL"
        print(f"  Case 11 (Radiation):      {status}")
    else:
        print(f"  Case 11 (Radiation):      ERROR - {r11['error']}")

    # Case 12
    r12 = all_results.get('case12_amr', {})
    if 'error' not in r12:
        n_init = r12.get('n_cells_initial', 0)
        n_final = r12.get('n_cells_final', 0)
        status = "PASS" if n_final > n_init else "FAIL"
        print(f"  Case 12 (AMR):            {status} ({n_init}->{n_final} cells)")
    else:
        print(f"  Case 12 (AMR):            ERROR - {r12['error']}")

    # Case 13
    r13 = all_results.get('case13_gpu', {})
    if 'error' not in r13:
        backend = r13.get('backend', 'unknown')
        acc = "PASS" if r13.get('accuracy_verified', False) else "FAIL"
        print(f"  Case 13 (GPU):            {acc} (backend={backend})")
    else:
        print(f"  Case 13 (GPU):            ERROR - {r13['error']}")

    # Case 14
    r14 = all_results.get('case14_3d_cavity', {})
    if 'error' not in r14:
        conv = r14.get('converged', False)
        phys = r14.get('physically_reasonable', False)
        status = "PASS" if (conv and phys) else "FAIL"
        print(f"  Case 14 (3D Cavity):      {status} (converged={conv}, physical={phys})")
    else:
        print(f"  Case 14 (3D Cavity):      ERROR - {r14['error']}")

    # Case 15
    r15 = all_results.get('case15_3d_convection', {})
    if 'error' not in r15:
        conv = r15.get('converged', False)
        strat = r15.get('stratified', False)
        status = "PASS" if (conv and strat) else "FAIL"
        nu = r15.get('nusselt', 0.0)
        print(f"  Case 15 (3D Convection):  {status} (Nu={nu:.4f}, stratified={strat})")
    else:
        print(f"  Case 15 (3D Convection):  ERROR - {r15['error']}")

    # Case 16
    r16 = all_results.get('case16_preconditioner', {})
    if 'error' not in r16:
        accurate = r16.get('all_accurate', False)
        status = "PASS" if accurate else "FAIL"
        best = r16.get('best_preconditioner', 'N/A')
        print(f"  Case 16 (Preconditioner): {status} (best={best})")
    else:
        print(f"  Case 16 (Preconditioner): ERROR - {r16['error']}")

    # Case 17
    r17 = all_results.get('case17_adaptive_dt', {})
    if 'error' not in r17:
        cfl_ok = r17.get('cfl_always_below_max', False)
        status = "PASS" if (r17.get('converged', False) and cfl_ok) else "FAIL"
        steps = r17.get('adaptive_dt_steps', 0)
        print(f"  Case 17 (Adaptive dt):    {status} (steps={steps}, CFL_ok={cfl_ok})")
    else:
        print(f"  Case 17 (Adaptive dt):    ERROR - {r17['error']}")

    # Case 18
    r18 = all_results.get('case18_web_dashboard', {})
    if 'error' not in r18:
        status = "PASS" if r18.get('converged', False) else "FAIL"
        pts = r18.get('n_data_points', 0)
        print(f"  Case 18 (Web Dashboard):  {status} (data_points={pts})")
    else:
        print(f"  Case 18 (Web Dashboard):  ERROR - {r18['error']}")

    # Case 19
    r19 = all_results.get('case19_boiling_condensation', {})
    if 'error' not in r19:
        status = "PASS" if r19.get('converged', False) else "FAIL"
        print(f"  Case 19 (Boiling/Cond):   {status}")
    else:
        print(f"  Case 19 (Boiling/Cond):   ERROR - {r19['error']}")

    # Case 20
    r20 = all_results.get('case20_edwards_blowdown', {})
    if 'error' not in r20:
        flash = r20.get('flashing_occurred', False)
        void = r20.get('max_void', 0.0)
        status = "PASS" if r20.get('converged', False) else "FAIL"
        print(f"  Case 20 (Edwards Flash):  {status} (flashing={flash}, max_void={void:.3f})")
    else:
        print(f"  Case 20 (Edwards Flash):  ERROR - {r20['error']}")

    # Case 21
    r21 = all_results.get('case21_6eq_heated_channel', {})
    if 'error' not in r21:
        conv21 = r21.get('converged', False)
        phys21 = r21.get('physical', False)
        status = "PASS" if (conv21 and phys21) else "FAIL"
        print(f"  Case 21 (6-Eq Channel):   {status} (converged={conv21}, physical={phys21})")
    else:
        print(f"  Case 21 (6-Eq Channel):   ERROR - {r21['error']}")

    # Case 22
    r22 = all_results.get('case22_hybrid_mesh', {})
    if 'error' not in r22:
        l2_22 = r22.get('L2_error', 1)
        status = "PASS" if l2_22 < 0.10 else "FAIL"
        print(f"  Case 22 (Hybrid Mesh):    {status} (L2={l2_22:.4e})")
    else:
        print(f"  Case 22 (Hybrid Mesh):    ERROR - {r22['error']}")

    # Case 23
    r23 = all_results.get('case23_3d_transient', {})
    if 'error' not in r23:
        l2_23 = r23.get('L2_error', 1)
        dev23 = r23.get('transient_development', False)
        status = "PASS" if r23.get('converged', False) else "FAIL"
        print(f"  Case 23 (3D Transient):   {status} (L2={l2_23:.4e}, dev={dev23})")
    else:
        print(f"  Case 23 (3D Transient):   ERROR - {r23['error']}")

    # Case 24
    r24 = all_results.get('case24_pool_boiling', {})
    if 'error' not in r24:
        boil = r24.get('boiling_occurred', False)
        ag24 = r24.get('max_alpha_g', 0)
        status = "PASS" if r24.get('converged', False) else "FAIL"
        print(f"  Case 24 (Pool Boiling):   {status} (boiling={boil}, max_ag={ag24:.4f})")
    else:
        print(f"  Case 24 (Pool Boiling):   ERROR - {r24['error']}")

    # Case 25
    r25 = all_results.get('case25_film_condensation', {})
    if 'error' not in r25:
        cond = r25.get('condensation_occurred', False)
        delta = r25.get('delta_numerical', 0)
        status = "PASS" if r25.get('converged', False) else "FAIL"
        print(f"  Case 25 (Condensation):   {status} (condensed={cond}, δ={delta*1000:.2f}mm)")
    else:
        print(f"  Case 25 (Condensation):   ERROR - {r25['error']}")

    print(f"\n  총 소요 시간: {total_elapsed:.1f}초")
    print("=" * 70)

    all_results['total_elapsed'] = total_elapsed
    return all_results


if __name__ == "__main__":
    results = run_all_cases()
