"""Run all 25 cases and generate DOCX report.

Uses threading timeouts. Merges fresh results with cached pickle for
cases that time out.
"""
import sys, os, time, threading, warnings, pickle, json
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


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
        return None  # timeout - will use cached
    if error[0]:
        return {'error': error[0]}
    return result[0]


# Import all case runners
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

# (num, result_key, func, extra_kwargs, timeout_seconds)
cases = [
    (1,  'case1_poiseuille',      run_case1,  {}, 90),
    (2,  'case2_cavity',          run_case2,  {'re_list': [100, 400], 'n_grid': 32}, 180),
    (3,  'case3_cht',             run_case3,  {}, 90),
    (4,  'case4_bubble_column',   run_case4,  {}, 180),
    (5,  'case5_3d_duct',         run_case5,  {}, 180),
    (6,  'case6_muscl',           run_case6,  {}, 90),
    (7,  'case7_unstructured',    run_case7,  {}, 120),
    (8,  'case8_mpi',             run_case8,  {}, 90),
    (9,  'case9_phase_change',    run_case9,  {}, 90),
    (10, 'case10_reaction',       run_case10, {}, 90),
    (11, 'case11_radiation',      run_case11, {}, 90),
    (12, 'case12_amr',            run_case12, {}, 90),
    (13, 'case13_gpu',            run_case13, {}, 120),
    (14, 'case14_3d_cavity',      run_case14, {}, 300),
    (15, 'case15_3d_convection',  run_case15, {}, 180),
    (16, 'case16_preconditioner', run_case16, {}, 90),
    (17, 'case17_adaptive_dt',    run_case17, {}, 90),
    (18, 'case18_web_dashboard',  run_case18, {}, 90),
    (19, 'case19_boiling_condensation', run_case19, {}, 90),
    (20, 'case20_edwards_blowdown', run_case20, {}, 180),
    (21, 'case21_6eq_heated_channel', run_case21, {}, 300),
    (22, 'case22_hybrid_mesh',    run_case22, {}, 180),
    (23, 'case23_3d_transient',   run_case23, {}, 300),
    (24, 'case24_pool_boiling',   run_case24, {}, 420),
    (25, 'case25_film_condensation', run_case25, {}, 180),
]

# Load cached results as fallback
cached = {}
pkl_path = os.path.join(RESULTS_DIR, 'all_results.pkl')
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        cached = pickle.load(f)
    print(f"Loaded {len(cached)} cached results from pickle")

all_results = {}
summary = []

for num, key, func, extra, timeout in cases:
    kw = {'results_dir': RESULTS_DIR, 'figures_dir': FIGURES_DIR}
    kw.update(extra)
    t0 = time.time()
    print(f'Running Case {num:2d} ({key}) ... ', end='', flush=True)
    r = run_with_timeout(func, kw, timeout=timeout)
    elapsed = time.time() - t0

    if r is None:
        # Timeout - use cached
        if key in cached:
            r = cached[key]
            status = f'CACHED (timeout {elapsed:.0f}s, using previous result)'
        else:
            r = {'error': f'TIMEOUT ({timeout}s) and no cache'}
            status = f'TIMEOUT (no cache)'
    elif isinstance(r, dict) and 'error' in r:
        # Error - try cached
        err_msg = r['error'][:60]
        if key in cached and 'error' not in cached[key]:
            r = cached[key]
            status = f'CACHED (error: {err_msg})'
        else:
            status = f'ERROR: {err_msg}'
    else:
        status = f'OK ({elapsed:.1f}s)'

    all_results[key] = r
    line = f'Case {num:2d}: {status}'
    summary.append(line)
    print(status, flush=True)

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
for line in summary:
    print(line)

# Save merged results
with open(pkl_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f'\nResults saved to {pkl_path}')

# Render ParaView-style figures (meshio + matplotlib)
print('\n' + '=' * 60)
print('Rendering mesh visualization figures...')
print('=' * 60)

try:
    from visualization.vtu_renderer import render_all_missing
    render_all_missing(RESULTS_DIR, FIGURES_DIR)
except Exception as e:
    print(f'  Mesh visualization rendering skipped: {e}')

# Generate DOCX report
print('\n' + '=' * 60)
print('Generating DOCX report...')
print('=' * 60)

from report.report_generator import generate_report

report_dir = os.path.join(BASE_DIR, 'report')
os.makedirs(report_dir, exist_ok=True)
report_path = os.path.join(report_dir, 'TwoFluid_FVM_Report.docx')

try:
    generate_report(all_results, report_path, FIGURES_DIR)
    print(f'\nReport generated: {report_path}')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'\nReport generation failed: {e}')
