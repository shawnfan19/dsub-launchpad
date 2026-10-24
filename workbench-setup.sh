#!/usr/bin/env bash
# Re-run on every new AoU Workbench Jupyter app (the boot disk starts blank). Safe to re-run.
set -euo pipefail

# dsub 0.5.0 side by side as dsub050/dstat050/ddel050 (Workbench's 0.5.2 breaks GPU jobs)
pip install -q pipx
pipx install dsub==0.5.0 --suffix=050   # no-op (exit 0) if already installed

# Patch it (idempotent): 0.5.0 pins cloud-sdk:294.0.0-slim for its helper runnables, which
# Google deleted 2026-09-01 (only rolling tags survive), and modern cloud-sdk is python3-only.
# Unpatched, every job exits 1 ~40s after RUNNING with no logs. Upstream: DataBiosphere/dsub#336.
p=$(ls -d "$(pipx environment --value PIPX_LOCAL_VENVS)"/dsub050/lib/python3*/site-packages/dsub/providers)
sed -i 's/cloud-sdk:294.0.0-slim/cloud-sdk:slim/' "$p/google_utils.py"
sed -i "s/| python -c '/| python3 -c '/" "$p/google_utils.py" "$p/google_batch.py"

# ponytail: editable-installs every attached workspace repo that is a Python package
for d in ~/repos/*/; do
  if [ -f "$d/pyproject.toml" ]; then pip install -q -e "$d"; fi
done

grep -qx 'alias submit=dsubmit' ~/.bashrc || echo 'alias submit=dsubmit' >> ~/.bashrc
