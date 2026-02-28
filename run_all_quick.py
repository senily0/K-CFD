"""Quick test runner - runs all 25 cases with timeout, prints summary."""
import sys, os, time, traceback, warnings, threading, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def run_with_timeout(func, kwargs, timeout=120):
    result = [None]
    error = [None]
    def target():
        try:
            result[0] = func(**kwargs)
        except Exception as e:
            error[0] = str(e)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return {'error': f'TIMEOUT ({timeout}s)'}
    if error[0]:
        return {'error': error[0]}
    return result[0] if result[0] else {'error': 'No result'}

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

cases = [
    (1, run_case1, {}, 60),
    (2, run_case2, {'re_list': [100, 400], 'n_grid': 32}, 120),
    (3, run_case3, {}, 60),
    (4, run_case4, {}, 120),
    (5, run_case5, {}, 120),
    (6, run_case6, {}, 60),
    (7, run_case7, {}, 60),
    (8, run_case8, {}, 60),
    (9, run_case9, {}, 60),
    (10, run_case10, {}, 60),
    (11, run_case11, {}, 60),
    (12, run_case12, {}, 60),
    (13, run_case13, {}, 60),
    (14, run_case14, {}, 180),
    (15, run_case15, {}, 120),
    (16, run_case16, {}, 60),
    (17, run_case17, {}, 60),
    (18, run_case18, {}, 60),
    (19, run_case19, {}, 60),
    (20, run_case20, {}, 60),
    (21, run_case21, {}, 180),
    (22, run_case22, {}, 120),
    (23, run_case23, {}, 180),
    (24, run_case24, {}, 300),
    (25, run_case25, {}, 120),
]

all_results = {}
summary = []

for num, func, extra, timeout in cases:
    kw = {'results_dir': 'results', 'figures_dir': 'figures'}
    kw.update(extra)
    t0 = time.time()
    r = run_with_timeout(func, kw, timeout=timeout)
    elapsed = time.time() - t0

    if isinstance(r, dict):
        if 'error' in r:
            status = f'ERROR: {r["error"][:80]}'
        elif r.get('L2_error') is not None:
            l2 = r['L2_error']
            status = f'PASS (L2={l2:.4e})' if l2 < 0.10 else f'FAIL (L2={l2:.4e})'
        elif 'converged' in r:
            status = 'PASS' if r.get('converged', False) else 'FAIL'
        elif 'physical_validity' in r:
            status = 'PASS' if r.get('physical_validity', False) else 'FAIL'
        elif 'physical' in r:
            status = 'PASS' if r.get('physical', False) else 'FAIL'
        elif 'accuracy_verified' in r:
            status = 'PASS' if r.get('accuracy_verified', False) else 'FAIL'
        elif 'n_cells_final' in r:
            status = 'PASS' if r.get('n_cells_final', 0) > r.get('n_cells_initial', 0) else 'FAIL'
        elif 'muscl_sharper' in r:
            status = 'PASS' if r.get('muscl_sharper', False) else 'FAIL'
        else:
            status = f'DONE (keys={list(r.keys())[:5]})'
    else:
        status = 'UNKNOWN'

    line = f'Case {num:2d}: {status} ({elapsed:.1f}s)'
    summary.append(line)
    all_results[f'case{num}'] = r
    # Flush immediately
    print(line, flush=True)

print('\n' + '='*60, flush=True)
print('SUMMARY', flush=True)
print('='*60, flush=True)
for line in summary:
    print(line, flush=True)

# Save results for report
with open('results/all_case_results.json', 'w') as f:
    # Convert numpy arrays etc to strings for JSON
    def convert(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    json.dump(convert(all_results), f, indent=2, default=str)

print('\nResults saved to results/all_case_results.json', flush=True)
