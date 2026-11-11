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
lines = []
for p in sorted(glob.glob(f'{OUT}/**/*', recursive=True)):
    if os.path.isfile(p) and os.path.basename(p) != 'SHA256SUMS.txt' and '__pycache__' not in p and not p.endswith('.pyc'): lines.append(hashlib.sha256(open(p, 'rb').read()).hexdigest() + '  ' + os.path.relpath(p, OUT))
open(f'{OUT}/SHA256SUMS.txt', 'w').write('\n'.join(lines) + '\n'); print(len(lines), 'files hashed')
