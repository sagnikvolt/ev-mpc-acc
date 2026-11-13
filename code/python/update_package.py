"""Adds the post-package work (regen-cap sensitivity, paper figure scripts) to open_data_package/ in place and rewrites SHA256SUMS.txt.
Does not move or delete anything. Run from anywhere: python3 python/update_package.py"""
import os, shutil, glob, hashlib
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/open_data_package'
for d in ['results/sens_cap50', 'code/paper_figures', 'results/figures']: os.makedirs(f'{OUT}/{d}', exist_ok=True)
for b in ['sens_cap50.py', 'sens_cap50_summary.py', 'update_package.py']: shutil.copy(f'{ROOT}/python/{b}', f'{OUT}/code/python/{b}')
shutil.copy(f'{ROOT}/paper/make_figures.py', f'{OUT}/code/paper_figures/make_figures.py')
shutil.copy(f'{ROOT}/admin/sens_cap50_prereg.md', f'{OUT}/preregistrations/sens_cap50_prereg.md')
for f in glob.glob(f'{ROOT}/results/sens_cap50/*'):
    b = os.path.basename(f)
    if os.path.getsize(f) > 8 and b != 'oracle_99.csv' and not b.endswith('.log'): shutil.copy(f, f'{OUT}/results/sens_cap50/{b}')   # skip empty/unused files
for f in glob.glob(f'{ROOT}/paper/figures/*'): shutil.copy(f, f'{OUT}/results/figures/{os.path.basename(f)}')

# --- Step E (2026-11-13): normal-car, safe-RL, feature importance, datasheet, eta 0.65 re-run, S1/S2 tables ---
import fnmatch
def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
for b in ['normal_car.py', 'safe_filter.py', 'safe_rl_gamma.py', 'safe_rl_train.py', 'summary_safe_rl.py', 'feature_importance.py', 'summary_ml_tables.py', 'datasheet.py',
          'rerun_sac_eta065.py', 'summary_sac_eta065.py', 'timing_solve.py', 'summary_eta065.py', 'rerun_eta065.py', 'rerun_eta065b.py', 'safe_rl_gamma.py']:
    if os.path.exists(f'{ROOT}/python/{b}'): cp(f'{ROOT}/python/{b}', f'{OUT}/code/python/{b}')
for b in ['run_safe_rl.bat', 'run_sac_eta065.bat']:
    if os.path.exists(f'{ROOT}/{b}'): cp(f'{ROOT}/{b}', f'{OUT}/code/{b}')
for b in ['normal_car_prereg.md', 'safe_rl_prereg.md', 'rerun_sac_eta065_prereg.md', 'rerun_eta065_prereg.md']:
    if os.path.exists(f'{ROOT}/open_data_package/preregistrations/{b}'): pass   # already in the package
for src in glob.glob(f'{ROOT}/results/normal_car/*') + glob.glob(f'{ROOT}/results/safe_rl/*'):
    if os.path.isfile(src): cp(src, f'{OUT}/results/{os.path.basename(os.path.dirname(src))}/{os.path.basename(src)}')
for src in glob.glob(f'{ROOT}/results/eta065_rerun/*'):
    if os.path.isfile(src) and os.path.getsize(src) > 0: cp(src, f'{OUT}/results/eta065_rerun/{os.path.basename(src)}')
for src in glob.glob(f'{ROOT}/results/ml/*'):
    b = os.path.basename(src)
    if os.path.isfile(src) and (b.startswith(('importance_', 'learning_curve', 'dataset_sizes', 'feature_importance_summary'))): cp(src, f'{OUT}/results/ml/{b}')
for b in ['datasheet_normal_vs_design.md', 'datasheet_normal_vs_design.csv']: cp(f'{ROOT}/results/{b}', f'{OUT}/results/{b}')
for b in ['scenarios_summary.txt', 's2_fixed_summary.txt']: cp(f'{ROOT}/results/gate3b/{b}', f'{OUT}/results/gate3b/{b}')
for f in glob.glob(f'{ROOT}/results/gate3b/s2_v1_archive/*'):
    if os.path.isfile(f): cp(f, f'{OUT}/results/gate3b/s2_v1_archive/{os.path.basename(f)}')
lines = []
for p in sorted(glob.glob(f'{OUT}/**/*', recursive=True)):
    if os.path.isfile(p) and os.path.basename(p) != 'SHA256SUMS.txt' and '__pycache__' not in p and not p.endswith('.pyc'): lines.append(hashlib.sha256(open(p, 'rb').read()).hexdigest() + '  ' + os.path.relpath(p, OUT))
open(f'{OUT}/SHA256SUMS.txt', 'w').write('\n'.join(lines) + '\n'); print(len(lines), 'files hashed')
