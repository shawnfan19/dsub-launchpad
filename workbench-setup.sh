#!/usr/bin/env bash
# Re-run on every new AoU Workbench Jupyter app (the boot disk starts blank). Safe to re-run.
set -euo pipefail

# dsub 0.5.0 side by side as dsub050/dstat050/ddel050 (Workbench's 0.5.2 breaks GPU jobs)
pip install -q pipx
pipx install dsub==0.5.0 --suffix=050   # no-op (exit 0) if already installed

# ponytail: editable-installs every attached workspace repo that is a Python package
for d in ~/repos/*/; do
  if [ -f "$d/pyproject.toml" ]; then pip install -q -e "$d"; fi
done

grep -qx 'alias submit=dsubmit' ~/.bashrc || echo 'alias submit=dsubmit' >> ~/.bashrc
