#!/usr/bin/env bash
# Builds a Lambda ZIP (python3.12, x86_64) WITHOUT Docker. Verified 2026-11-12: 166 MB unzipped (limit 250 MB), handler runs on CPython 3.12.
# The scs solver is left out on purpose (106+39 MB; the code uses OSQP only). numpy/_core/tests must be kept (numpy imports it).
set -e; HERE=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$HERE"); W=${1:-/tmp/evzip}; rm -rf "$W"; mkdir -p "$W/wh" "$W/b"
pip download -q --dest "$W/wh" --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
  --python-version 3.12 --implementation cp -r "$HERE/app/requirements.txt"
for w in "$W"/wh/*.whl; do case $w in *scs-*|*setuptools*) continue;; esac; python3 -m zipfile -e "$w" "$W/b/"; done
cd "$W/b"; find . -type d -name __pycache__ -prune -exec rm -rf {} +; find . -type d -name tests -not -path "./numpy/_core/tests" -prune -exec rm -rf {} +
cp "$HERE/app/handler.py" .; mkdir -p code/python derived_data; cp "$PKG/code/python/mpc_acc.py" "$PKG/code/python/baselines.py" code/python/; cp "$PKG/derived_data/bmw_i3_params_final.csv" derived_data/
python3 -c "import shutil; shutil.make_archive('$W/ev-demo-lambda', 'zip', '.')"; du -sh "$W/b" "$W/ev-demo-lambda.zip"
